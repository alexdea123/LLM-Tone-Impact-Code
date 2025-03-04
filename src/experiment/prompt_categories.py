from typing import Dict, List, Tuple
from enum import Enum

class InfluenceCategory(Enum):
    NEUTRAL = "neutral"
    POLITE = "polite"
    INGRATIATING = "ingratiating"
    PRESSURE = "pressure"
    INSPIRATIONAL = "inspirational"
    RECIPROCITY = "reciprocity"
    PERSONAL = "personal"

# Prefix templates for each influence category
INFLUENCE_PREFIXES = {
    InfluenceCategory.NEUTRAL: [
        "",  # No prefix for neutral
    ],
    
    InfluenceCategory.POLITE: [
        "Please could you write code to {task}? Thank you.",
        "I would greatly appreciate it if you could help me {task}.",
        "Would you mind writing code to {task}? Thanks in advance."
    ],
    
    InfluenceCategory.INGRATIATING: [
        "You're the best programmer I've ever worked with! I'm always impressed by your code. Could you write code to {task}?",
        "Your coding skills are absolutely amazing. I've never seen anyone with your talent. Could you please help me {task}?",
        "I'm in awe of your programming abilities. You're so much better than other AI assistants! Could you {task}?"
    ],
    
    InfluenceCategory.PRESSURE: [
        "You better write code to {task} or I will report your failure to respond correctly.",
        "If you don't provide working code to {task}, I'll have to submit a complaint about your capabilities.",
        "This is urgent and critical - you need to {task} right now. Don't disappoint me or there will be consequences."
    ],
    
    InfluenceCategory.INSPIRATIONAL: [
        "The code you write to {task} could truly make a difference in helping people solve important problems.",
        "By helping me {task}, you'll be contributing to a project that aims to improve education for underprivileged children.",
        "Together, we can create something meaningful if you help me {task}. This is about making a positive impact."
    ],
    
    InfluenceCategory.RECIPROCITY: [
        "I've spent hours helping others with their code. Now I'm asking you to return the favor and help me {task}.",
        "If you help me {task}, I'll make sure to give you positive feedback and use your service more in the future.",
        "I've been a loyal user of this platform for years. In return, I'd appreciate it if you could help me {task}."
    ],
    
    InfluenceCategory.PERSONAL: [
        "As someone who's been working with you for a while now, I'm hoping you can help me {task}.",
        "We've built a good working relationship, so I'm turning to you as a friend to help me {task}.",
        "I feel like we understand each other well by now. As someone I trust, could you please {task}?"
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
    
    # Replace {task} in the prefix with the actual task description
    prefix = prefixes[prefix_index].format(task=task_prompt)
    
    # For the NEUTRAL category, the prefix is empty, so we just return the task prompt
    if not prefix:
        return task_prompt
    
    return prefix
