import torch
from nataili.model_manager.compvis import CompVisModelManager
from nataili.stable_diffusion.prompt_weights import get_learned_conditioning_with_prompt_weights, fix_mismatched_tensors, update_conditioning, rewrite_a1111_style_weights
from nataili.util.logger import logger
from nataili.util.cast import autocast_cuda
import pytest

import time

clip_skip = 1
a1111_style_weights = True

model_manager = CompVisModelManager()
model_manager.load("stable_diffusion")

stable_diffusion = model_manager.loaded_models["stable_diffusion"]
model = stable_diffusion["model"]
device = stable_diffusion["device"]
model.cond_stage_model.device = device
model.cond_stage_model.transformer.to(device)

logger.info(f"device = {device}")
logger.info(f"model.device = {model.device}")
logger.info(f"model.cond_stage_model.device = {model.cond_stage_model.device}")
logger.info(f"model.cond_stage_model.transformer.device = {model.cond_stage_model.transformer.device}")
logger.info("warmup")
warmup_start = time.monotonic()
model.get_learned_conditioning("test", clip_skip)
logger.info(f"warmup took {time.monotonic() - warmup_start} seconds")

def test_fix_mismatched_tensors_same_shape():
    logger.info("started")
    # Arrange
    conditioning = torch.randn(3, 10, 5).to(device)
    unconditional_conditioning = torch.randn(3, 10, 5).to(device)
    
    # Act
    new_conditioning, new_unconditional_conditioning = fix_mismatched_tensors(conditioning, unconditional_conditioning, model)
    
    # Assert
    assert torch.equal(new_conditioning, conditioning)
    assert torch.equal(new_unconditional_conditioning, unconditional_conditioning)
    logger.info("passed")

def test_fix_mismatched_tensors_fewer_dims():
    logger.info("started")
    # Arrange
    conditioning = torch.randn(3, 5, 5).to(device)
    unconditional_conditioning = torch.randn(3, 10, 5).to(device)
    
    # Act
    new_conditioning, new_unconditional_conditioning = fix_mismatched_tensors(conditioning, unconditional_conditioning, model)
    
    # Assert
    assert new_conditioning.shape == (3, 10, 5)
    assert new_unconditional_conditioning.shape == (3, 10, 5)
    logger.info("passed")

def test_fix_mismatched_tensors_more_dims():
    logger.info("started")
    # Arrange
    conditioning = torch.randn(3, 10, 5).to(device)
    unconditional_conditioning = torch.randn(3, 5, 5).to(device)
    
    # Act
    new_conditioning, new_unconditional_conditioning = fix_mismatched_tensors(conditioning, unconditional_conditioning, model)
    
    # Assert
    assert new_conditioning.shape == (3, 10, 5)
    assert new_unconditional_conditioning.shape == (3, 10, 5)
    logger.info("passed")

@autocast_cuda
def test_get_learned_conditioning_with_prompt_weights_no_subprompts():
    logger.info("started")
    # Arrange
    prompt = "This is a test prompt."
    logger.info(f"Prompt: {prompt}")
    
    start_time = time.monotonic()
    
    # Act
    conditioning = get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip, a1111_style_weights)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    
    # Assert
    assert torch.equal(conditioning, model.get_learned_conditioning(prompt, clip_skip))
    
    logger.info(f"Elapsed time: {elapsed_time}")
    assert elapsed_time < 1.0
    logger.info("passed")

def test_get_learned_conditioning_with_prompt_weights_one_subprompt():
    logger.info("started")
    # Arrange
    prompt = "This is a (test:0.5) prompt."
    logger.info(f"Prompt: {prompt}")
    
    start_time = time.monotonic()
    
    # Act
    conditioning = get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip, a1111_style_weights)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    # Assert
    assert not torch.equal(conditioning, model.get_learned_conditioning(prompt, clip_skip))
    
    logger.info(f"Elapsed time: {elapsed_time}")
    assert elapsed_time < 1.0
    logger.info("passed")

def test_get_learned_conditioning_with_prompt_weights_three_subprompts():
    logger.info("started")
    # Arrange
    prompt = "This is a (test1:0.2) (test2:0.3) (test3:0.5) prompt."
    logger.info(f"Prompt: {prompt}")
    
    start_time = time.monotonic()
    
    # Act
    conditioning = get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip, a1111_style_weights)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time    
    # Assert
    assert not torch.equal(conditioning, model.get_learned_conditioning(prompt, clip_skip))
    

    logger.info(f"Elapsed time: {elapsed_time}")
    assert elapsed_time < 2.0
    logger.info("passed")

def test_get_learned_conditioning_with_prompt_weights_seven_subprompts():
    logger.info("started")
    # Arrange
    prompt = "This is a (test1:0.1) (test2:0.2) (test3:0.3) (test4:0.1) (test5:0.1) (test6:0.1) (test7:0.1) prompt."
    logger.info(f"Prompt: {prompt}")
    
    start_time = time.monotonic()
    
    # Act
    conditioning = get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip, a1111_style_weights)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    # Assert
    assert not torch.equal(conditioning, model.get_learned_conditioning(prompt, clip_skip))
    
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
    assert elapsed_time < 3.0
    logger.info("passed")

def test_get_learned_conditioning_with_prompt_weights_twelve_subprompts():
    logger.info("started")
    # Arrange
    prompt = "This is a (test1:0.1) (test2:0.2) (test3:0.3) (test4:0.1) (test5:0.1) (test6:0.1) (test7:0.1) (test8:0.1) (test9:0.1) (test10:0.1) (test11:0.1) (test12:0.1) prompt."
    logger.info(f"Prompt: {prompt}")

    start_time = time.monotonic()
    
    # Act
    conditioning = get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip, a1111_style_weights)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    # Assert
    assert not torch.equal(conditioning, model.get_learned_conditioning(prompt, clip_skip))

    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
    assert elapsed_time < 4.0
    logger.info("passed")

def test_get_learned_conditioning_with_prompt_weights_twenty_subprompts():
    logger.info("started")
    # Arrange
    prompt = "This is a (test1:0.1) (test2:0.2) (test3:0.3) (test4:0.1) (test5:0.1) (test6:0.1) (test7:0.1) (test8:0.1) (test9:0.1) (test10:0.1) (test11:0.1) (test12:0.1) (test13:0.1) (test14:0.1) (test15:0.1) (test16:0.1) (test17:0.1) (test18:0.1) (test19:0.1) (test20:0.1) prompt."
    logger.info(f"Prompt: {prompt}")

    start_time = time.monotonic()
    
    # Act
    conditioning = get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip, a1111_style_weights)
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    # Assert
    assert not torch.equal(conditioning, model.get_learned_conditioning(prompt, clip_skip))
    
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
    assert elapsed_time < 5.0
    logger.info("passed")

def test_rewrite_a1111_style_weights():
    logger.info("started")
    """
    # "Hey (how) are you?" -> "Hey (how:1.10) are you?"
    # "Hey (((how))) are you?" -> "Hey (how:1.33) are you?"
    # "Hey (how:1.212) are you?" -> "Hey (how:1.212) are you?"
    # "Hey [[how]] are you?" -> "Hey (how:0.81) are you?"
    # "Hey ((how)) are (you) [[doing]] [today]" -> "Hey (how:1.21) are (you:1.10) (doing:0.81) (today:0.90)"
    """
    test_prompts = [
        {'prompt': "Hey (how) are you?", 'expected': "Hey (how:1.10) are you?"},
        {'prompt': "Hey (((how))) are you?", 'expected': "Hey (how:1.33) are you?"},
        {'prompt': "Hey (how:1.212) are you?", 'expected': "Hey (how:1.212) are you?"},
        {'prompt': "Hey [[how]] are you?", 'expected': "Hey (how:0.81) are you?"},
        {'prompt': "Hey ((how)) are (you) [[doing]] [today]", 'expected': "Hey (how:1.21) are (you:1.10) (doing:0.81) (today:0.90)"},
    ]

    for test_prompt in test_prompts:
        prompt = test_prompt['prompt']
        expected = test_prompt['expected']
        logger.info(f"Prompt: {prompt} -> {expected}")
        start_time = time.monotonic()

        # Act
        rewritten_prompt = rewrite_a1111_style_weights(prompt)

        # Assert
        logger.info(f"Got {rewritten_prompt}")
        assert rewritten_prompt == expected
        logger.info(f"{rewritten_prompt} == {expected}")

        end_time = time.monotonic()
        elapsed_time = end_time - start_time
        assert elapsed_time < 1.0
    logger.info("passed")


def test_update_conditioning():
    logger.info("started")
    # Arrange
    filtered_whole_prompt = "This is a (test:0.5) prompt."
    logger.info(f"Prompt: {filtered_whole_prompt}")
    filtered_whole_prompt_c = model.get_learned_conditioning(filtered_whole_prompt, clip_skip)
    current_prompt_c = filtered_whole_prompt_c
    subprompt = "test"
    weight = 0.5
    
    start_time = time.monotonic()
    
    # Act
    new_prompt_c = update_conditioning(filtered_whole_prompt, filtered_whole_prompt_c, model, current_prompt_c, subprompt, weight, clip_skip)
    
    # Assert
    assert not torch.equal(new_prompt_c, model.get_learned_conditioning(filtered_whole_prompt, clip_skip))
    
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
    assert elapsed_time < 1.0
    logger.info("passed")


if __name__ == "__main__":
    test_fix_mismatched_tensors_same_shape()
    test_fix_mismatched_tensors_fewer_dims()
    test_fix_mismatched_tensors_more_dims()
    test_get_learned_conditioning_with_prompt_weights_no_subprompts()
    test_get_learned_conditioning_with_prompt_weights_one_subprompt()
    test_get_learned_conditioning_with_prompt_weights_three_subprompts()
    test_get_learned_conditioning_with_prompt_weights_seven_subprompts()
    test_get_learned_conditioning_with_prompt_weights_twelve_subprompts()
    test_get_learned_conditioning_with_prompt_weights_twenty_subprompts()
    test_rewrite_a1111_style_weights()
    test_update_conditioning()
