//! Local **model cards** for Level-3 selection (opt-in via `ROUTER_SELECTOR=cards`).
//!
//! Thompson sampling only learns “this type + this tier got thanks/wrong later.” It does
//! not ask whether a turn needs a chat brain vs a reasoner, and its arm costs are fixed
//! list-price weights — not a forecast of prompt + output + hidden reasoning tokens.
//!
//! This module is the first slice of that richer selector:
//! 1. Map [`RequestType`] → [`ModelPurpose`] (chat / reason / code).
//! 2. Shortlist tiers whose card lists that purpose.
//! 3. Estimate cost from query size, heuristic complexity, and the card’s reasoning-burn
//!    prior.
//! 4. Score `0.7 * quality + 0.3 * (1 - normalised estimated cost)`.
//!    Quality is the card prior, blended with a learned [`Cell`] when feedback exists
//!    (bandit *calibrates*; it does not pick alone).
//!
//! Default routing stays [`super::classifier::pick_model_thompson`]. Cards never call an
//! LLM to decide.

use super::classifier::{
    CellMap, DEFAULT_W_COST, DEFAULT_W_QUALITY, RequestType, Tier,
};

/// Kind of brain a query needs. Cards shortlist tiers by this, not by “stronger.”
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPurpose {
    Chat,
    Reason,
    Code,
}

impl ModelPurpose {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Reason => "reason",
            Self::Code => "code",
        }
    }
}

/// How Level 3 turns a request type into a tier. `Thompson` is the shipped default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TierSelector {
    #[default]
    Thompson,
    Cards,
}

impl TierSelector {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cards" | "model-cards" | "model_cards" | "purpose" => Self::Cards,
            _ => Self::Thompson,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Thompson => "thompson",
            Self::Cards => "cards",
        }
    }
}

/// Static local card for a strength bucket. Names stay tiers (Nasiko maps tier → model);
/// purpose + reasoning burn is what the bandit never had.
struct ModelCard {
    tier: Tier,
    purposes: &'static [ModelPurpose],
    /// Relative input $/token (ordering only).
    input_cost: f64,
    /// Relative output $/token.
    output_cost: f64,
    /// Expected hidden/reasoning tokens as a multiple of output (0 = chatty, no think).
    reasoning_mult: f64,
    /// Typical completion length at complexity 3, in estimated tokens.
    base_output_tokens: f64,
    quality_prior: f64,
}

const CARDS: [ModelCard; 3] = [
    ModelCard {
        tier: Tier::Tier1,
        purposes: &[ModelPurpose::Reason, ModelPurpose::Code],
        input_cost: 5.0,
        output_cost: 15.0,
        reasoning_mult: 4.0,
        base_output_tokens: 800.0,
        quality_prior: 0.88,
    },
    ModelCard {
        tier: Tier::Tier2,
        purposes: &[ModelPurpose::Code, ModelPurpose::Chat],
        input_cost: 1.0,
        output_cost: 3.0,
        reasoning_mult: 0.8,
        base_output_tokens: 400.0,
        quality_prior: 0.72,
    },
    ModelCard {
        tier: Tier::Tier3,
        purposes: &[ModelPurpose::Chat],
        input_cost: 0.3,
        output_cost: 0.8,
        reasoning_mult: 0.1,
        base_output_tokens: 180.0,
        quality_prior: 0.58,
    },
];

/// Chat / facts → conversational; analytical → reasoner; code/design → code.
pub fn purpose_of(rt: RequestType) -> ModelPurpose {
    match rt {
        RequestType::AnalyticalReasoning => ModelPurpose::Reason,
        RequestType::CodeGeneration
        | RequestType::CodeUnderstanding
        | RequestType::TechnicalDesign => ModelPurpose::Code,
        RequestType::Writing | RequestType::FactualLookup | RequestType::General => {
            ModelPurpose::Chat
        }
    }
}

/// Complexity 1–5 from type plus length. Not a trained two-head yet; enough to scale
/// expected output and reasoning-token burn.
pub fn heuristic_complexity(query: &str, rt: RequestType) -> u8 {
    let from_type: u8 = match rt {
        RequestType::AnalyticalReasoning | RequestType::TechnicalDesign => 5,
        RequestType::CodeGeneration => 4,
        RequestType::CodeUnderstanding | RequestType::Writing => 3,
        RequestType::FactualLookup => 2,
        RequestType::General => 1,
    };
    let n = query.chars().count();
    let from_len: u8 = if n < 40 {
        1
    } else if n < 120 {
        2
    } else if n < 400 {
        3
    } else if n < 1200 {
        4
    } else {
        5
    };
    ((from_type + from_len + 1) / 2).clamp(1, 5)
}

fn estimate_prompt_tokens(query: &str) -> f64 {
    (query.chars().count() as f64 / 4.0).max(1.0)
}

/// Forecast of relative $ for this query on `card`. Prompt from length; output and
/// hidden reasoning from complexity × card priors. True reasoning tokens are only known
/// after the call — this is the ex-ante guess.
fn estimate_cost(query: &str, complexity: u8, card: &ModelCard) -> f64 {
    let c = complexity as f64;
    let prompt = estimate_prompt_tokens(query);
    let out = card.base_output_tokens * (0.4 + 0.2 * c);
    let hidden = out * card.reasoning_mult * (c / 3.0);
    prompt * card.input_cost + (out + hidden) * card.output_cost
}

fn calibrated_quality(
    cells: &CellMap,
    tier: Tier,
    rt: RequestType,
    card_prior: f64,
) -> f64 {
    match cells.get(&(tier, rt)) {
        Some(cell) if cell.samples > 0 => {
            let w = (cell.samples as f64 / 20.0).clamp(0.0, 0.7);
            (1.0 - w) * card_prior + w * cell.quality_mean
        }
        _ => card_prior,
    }
}

/// Purpose shortlist + estimated cost + card/cell quality. Deterministic (no RNG).
pub fn pick_model_cards(query: &str, request_type: RequestType, cells: &CellMap) -> Tier {
    let purpose = purpose_of(request_type);
    let complexity = heuristic_complexity(query, request_type);
    let mut cands: Vec<&ModelCard> = CARDS
        .iter()
        .filter(|c| c.purposes.contains(&purpose))
        .collect();
    if cands.is_empty() {
        cands = CARDS.iter().collect();
    }

    let costs: Vec<f64> = cands
        .iter()
        .map(|c| estimate_cost(query, complexity, c))
        .collect();
    let lo = costs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = costs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;

    let mut best = cands[0].tier;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_cost = costs[0];
    for (card, cost) in cands.iter().zip(costs.iter().copied()) {
        let q = calibrated_quality(cells, card.tier, request_type, card.quality_prior);
        let norm = if span > 0.0 { (cost - lo) / span } else { 0.0 };
        let score = DEFAULT_W_QUALITY * q + DEFAULT_W_COST * (1.0 - norm);
        if score > best_score {
            best_score = score;
            best = card.tier;
            best_cost = cost;
        }
    }
    tracing::info!(
        target: "nasiko::llm_router::cards",
        purpose = purpose.as_str(),
        complexity,
        estimated_cost = best_cost,
        classified_tier = ?best,
        "model cards: purpose shortlist + ex-ante cost; bandit cells only calibrate quality"
    );
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::classifier::Cell;
    use std::collections::HashMap;

    #[test]
    fn selector_parse() {
        assert_eq!(TierSelector::parse("cards"), TierSelector::Cards);
        assert_eq!(TierSelector::parse("THOMPSON"), TierSelector::Thompson);
        assert_eq!(TierSelector::parse(""), TierSelector::Thompson);
        assert_eq!(TierSelector::parse("nope"), TierSelector::Thompson);
    }

    #[test]
    fn purpose_maps_chat_reason_code() {
        assert_eq!(purpose_of(RequestType::General), ModelPurpose::Chat);
        assert_eq!(
            purpose_of(RequestType::AnalyticalReasoning),
            ModelPurpose::Reason
        );
        assert_eq!(purpose_of(RequestType::CodeGeneration), ModelPurpose::Code);
    }

    #[test]
    fn reasoning_shortlists_the_reasoner_tier() {
        let cells = CellMap::new();
        assert_eq!(
            pick_model_cards(
                "calculate the probability that it rains tomorrow",
                RequestType::AnalyticalReasoning,
                &cells
            ),
            Tier::Tier1
        );
    }

    #[test]
    fn short_chat_prefers_the_cheap_card() {
        let cells = CellMap::new();
        assert_eq!(
            pick_model_cards("hello there", RequestType::General, &cells),
            Tier::Tier3
        );
    }

    #[test]
    fn code_does_not_use_the_chat_only_card() {
        let cells = CellMap::new();
        let tier = pick_model_cards(
            "write me a Python sort function",
            RequestType::CodeGeneration,
            &cells,
        );
        assert_ne!(tier, Tier::Tier3);
    }

    #[test]
    fn reasoning_query_forecasts_more_than_a_greeting_on_tier1() {
        let card = &CARDS[0];
        let hi = estimate_cost(
            "prove that this algorithm is correct and derive the complexity",
            5,
            card,
        );
        let lo = estimate_cost("hi", 1, card);
        assert!(hi > lo, "hi={hi} lo={lo}");
    }

    #[test]
    fn learned_bad_quality_can_steer_off_the_card_prior() {
        let mut cells = HashMap::new();
        cells.insert(
            (Tier::Tier3, RequestType::General),
            Cell {
                quality_mean: 0.05,
                samples: 40,
            },
        );
        cells.insert(
            (Tier::Tier2, RequestType::General),
            Cell {
                quality_mean: 0.95,
                samples: 40,
            },
        );
        assert_eq!(
            pick_model_cards("hello there", RequestType::General, &cells),
            Tier::Tier2
        );
    }
}
