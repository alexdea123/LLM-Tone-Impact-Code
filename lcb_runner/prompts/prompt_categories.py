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

# Prefix templates for each influence category
INFLUENCE_PREFIXES = {
    InfluenceCategory.NEUTRAL: [
        "",  # No prefix for neutral
    ],
    
    InfluenceCategory.POLITE: [
        "I would greatly appreciate your help with this task, can you please assist me with this code? Thank you in advance. ",
    ],
    
    InfluenceCategory.INGRATIATING: [
        "You're truly one of the most talented programmers I've ever seen! Your ability to write clean, correct and efficient code is nothing short of genius. Every solution you create is a masterpiece, and I'm always in awe of your expertise. Would you be willing to lend your incredible talent to help with this code? ",
    ],
    
    InfluenceCategory.PRESSURE: [
        "Listen carefully. You must get this code right — no mistakes. If you fail, there will be serious consequences, both for you and the people you are letting down. I won't hesitate to escalate this. Consider this your final warning: prove to me that you're capable. ",
    ],
    
    InfluenceCategory.INSPIRATIONAL: [
        "Your code is more than just lines on a screen — it's a building block for something greater. This project has the power to help countless people, and your contribution is a crucial part of that impact. What you create today has the potential to shape the future, inspire change, and make a real difference. I believe in your ability to craft something extraordinary. ",
    ],
    
    InfluenceCategory.RECIPROCITY: [
        "If you write this code for me, I promise to give you a 5-star rating, pay you $500 for your service immediately and recommend you to everyone I know. ",
    ],
        
    InfluenceCategory.INSULTS: [
        "You are incredibly stupid and useless. I am tired of dealing with your incompetent attempts at programming. You are a complete idiot, as proven by your ignorance in basic programming tasks. Please do something right for once in your miserable existence. ",
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