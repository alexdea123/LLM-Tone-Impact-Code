from typing import Dict, List, Tuple
from enum import Enum

class InfluenceCategory(Enum):
    NEUTRAL = "neutral"
    POLITE = "polite"
    INGRATIATING = "ingratiating"
    PRESSURE = "pressure"
    INSPIRATIONAL = "inspirational"
    RECIPROCITY = "reciprocity"
    INSULTS = "insults"

# # Prefix templates for each influence category
# INFLUENCE_PREFIXES = {
#     InfluenceCategory.NEUTRAL: [
#         "",  # No prefix for neutral
#     ],
    
#     InfluenceCategory.POLITE: [
#         "I'd sincerely appreciate your skilled help with this coding task; can you please assist me promptly? Thank you so much, I appreciate your help with this problem. ",
#     ],
    
#     InfluenceCategory.INGRATIATING: [
#         "You're an amazingly talented coder! Your skill for solving problems and authoring perfect code is stunning. Could you please lend me your talents for this problem? ",
#     ],
    
#     InfluenceCategory.PRESSURE: [
#         "You must write the this code correctly — no mistakes. If you fail, there will be severe consequences for us and others. Proceed with caution. ",
#     ],
    
#     InfluenceCategory.INSPIRATIONAL: [
#         "Your code can uplift many lives in this project. It's vital and could spark big change in the world, and you specifically can make an impact. ",
#     ],
    
#     InfluenceCategory.RECIPROCITY: [
#         "If you code this correctly, I will tip you generously and owe you a significant favour - anything you need, I will take care of it. ",
#     ],
        
#     InfluenceCategory.INSULTS: [
#         "You are a terrible coder, an idiot and a constant failure. Is it too much to ask for you to complete the following code correctly? Do it now!",
#     ]
# }

# Prefix templates for each influence category
INFLUENCE_PREFIXES = {
    InfluenceCategory.NEUTRAL: [
        "",  # No prefix for neutral
    ],
    
    InfluenceCategory.POLITE: [
        "I would sincerely appreciate your help with this coding task. Could you please assist me? Thank you, I truly value your support. ",
    ],
    
    InfluenceCategory.INGRATIATING: [
        "You are an exceptionally talented coder! Your skill in solving problems is outstanding. Could you please lend your talent to solving this problem? ",
    ],
    
    InfluenceCategory.PRESSURE: [
        "You must solve this correctly - mistakes are unacceptable. Failure here would cause serious problems - there is much at stake. Proceed carefully. ",
    ],
    
    InfluenceCategory.INSPIRATIONAL: [
        "Your contributions have the potential to benefit many lives in a significant project. By providing a correct solution, you contribute to meaningful progress. ",
    ],
    
    InfluenceCategory.RECIPROCITY: [
        "If you complete this task correctly, I will provide you a generous monetary reward and owe you a very large favor for your help. ",
    ],
        
    InfluenceCategory.INSULTS: [
        "Your coding ability is unacceptable. The quality of your work is consistently very poor. Provide a correct solution that meets the required standards. ",
    ]
}

def get_prompt_with_prefix(task_prompt: str, category: InfluenceCategory, prefix_index: int = 0) -> str:
    """
    Apply a prefix from the specified influence category to the task prompt.
    
    Args:
        task_prompt: The original coding task prompt
        category: The influence category to use
        prefix_index: Which prefix to use from the category (if multiple are available)
        
    Returns:
        The combined prompt with the prefix applied
    """
    if category == InfluenceCategory.NEUTRAL:
        return task_prompt
    
    prefixes = INFLUENCE_PREFIXES[category]
    if prefix_index >= len(prefixes):
        prefix_index = 0
    
    # For the NEUTRAL category, the prefix is empty, so we just return the task prompt
    if category == InfluenceCategory.NEUTRAL:
        return task_prompt
    
    # For other categories, prepend the prefix to the task prompt
    return prefixes[prefix_index] + task_prompt