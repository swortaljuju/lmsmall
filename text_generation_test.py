import os
import torch
from torch.nn import functional as F
from baseline_gpt2 import model_name as gp2_model_name, GPTConfig, GPT
from conv_attention import (
    model_name as conv_model_name,
    Config as ConvConfig,
    ConvAttention,
)
from progressive_diverging import (
    model_name as prog_model_name,
    Config as ProgConfig,
    ProgressiveDiverging,
)
from base_trainer import BaseTrainer
import tiktoken
import time

TEST_PROMPTS = [
    """Write an educational piece suited for college students related to the following text snippet: "# logical implication ## 1 Short version Logical implication is an operation on two logical values, typically the values of two propositions (http://planetmath.org/PropositionalCalculus), that produces a value of false just in case the first operand is true and the second operand is false. The truth table for the logical implication operation that is written as $p\\Rightarrow q$ and read as $p\\ \\operatorname{implies}\\ q\\",$ also written as $p\\rightarrow q$ and read as $\\operatorname{if}\\ p\\ \\operatorname{then}\\ q",$ is as follows: Logical Implication $p$ $q$ $p\\Rightarrow q$ F F T F T T T F F T T T ## 2 Long version The mathematical objects that inform our capacity for logical reasoning are easier to describe in a straightforward way than it is to reconcile their traditional accounts. Still, some discussion of the language that occurs in the literature cannot be avoided. The concept of logical implication encompasses a specific logical function, a specific logical relation, and t" Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.""",
    """Write an educational piece suited for college students related to the following text snippet: "# How do you find (d^2y)/(dx^2) for 3x^2+y^2=2? Feb 19, 2017 $\\frac{{d}^{2} y}{{\\mathrm{dx}}^{2}} = - \\frac{6}{y} ^ 3$ #### Explanation: When we differentiate $y$, we get $\\frac{\\mathrm{dy}}{\\mathrm{dx}}$ When we differentiate a non implicit function of $y$ then using the chain rule we can differentiate wrt $x$ and multiply by $\\frac{\\mathrm{dy}}{\\mathrm{dx}}$. When this is done in situ it is known as implicit differentiation. We have: $3 {x}^{2} + {y}^{2} = 2$ Differentiate wrt $x$ and we get: $\\setminus \\setminus \\setminus 6 x + 2 y \\frac{\\mathrm{dy}}{\\mathrm{dx}} = 0$ $\\therefore 3 x + y \\frac{\\mathrm{dy}}{\\mathrm{dx}} = 0$ And now (as we want the second derivative) we differentiate wrt x again, this time we must also apply the product rule: $3 + \\left(y\\right) \\left(\\frac{{d}^{2} y}{{\\mathrm{dx}}^{2}}\\right) + \\left(\\frac{\\mathrm{dy}}{\\mathrm{dx}}\\right) \\left(\\frac{\\mathrm{dy}}{\\mathrm{dx}}\\right) = 0$ $\\therefore 3 + y \\frac{{d}^{2} y}{{\\mathrm{dx}}^{2}} + {\\left(\\frac{" Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.""",
    """Write an educational piece suited for college students related to the following text snippet: "NILAKANTHA SERIES PROOF Please Sign up or sign in to vote. Ranjan Roy, Mathematics Magazine , Vol. Historically, one of the best approximations of PI and interestingly also one of the oldest, was used by the Chinese mathematician Zu Chongzhi Sec. The most recent calculation found more than 13 trillion digits of pi in days! The article was complemented according to your suggestion. Not impressed Jose A Pascoa 8-Aug 7: The first written description of an infinite series that could be used to compute pi was laid out in Sanskrit verse by Indian astronomer Nilakantha Somayaji around A. Now pick up a pen, close your eyes and put dots on the square at random. Articles Quick Answers Messages. I had to raise the iteration count way up to get it to register with the timer. I leave the conclusion to you when examining the table above. PI with decimals: Juan Manuel Romero Martin 4-Sep 9: PI with decimals: If you do this enough times, and your efforts are truly random, eventually the percentage " Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.""",
    """Write an educational piece suited for college students related to the following text snippet: "# Proof a Weyl Algebra isn't isomorphic to a matrix ring over a division ring Can anyone prove that a Weyl Algebra is not isomorphic to a matrix ring over a division ring? - Notation: The Weyl algebra is $$k[x_1, x_2, \\ldots, x_n, \\partial_1, \\partial_2, \\ldots, \\partial_n]$$ with the obvious relations. The Weyl algebra doesn't contain any division rings larger than $k$, and it is infinite dimensional over $k$. So, assuming you don't allow infinite matrices, that's a proof. How to see that it doesn't contain any division ring larger than $k$? I just need to show that any nonconstant differential operator is not invertible. One way to see this is to notice that multiplying differential operators multiplies symbols, and the symbol of a nonconstant differential operator is a nonconstant polynomial. - Whats the base field? – Casebash Nov 8 '09 at 4:42 The field of constants. I have edited the answer; see if that is clearer. – David Speyer Nov 8 '09 at 13:49 I guess you could also use" Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.""",
    """Write an educational piece suited for college students related to the following text snippet: "# Ordered field, Bounded set, and the containment I am now in engineering mathematics class and it goes over some basic set theory. Since I haven't had any experience with set theory, three statements leave me confused. Thanks for your help! Statement: 1) General comment (not theorem or axiom): Set > Group > Ring > Field 2) Imagine set like set of Natural number is not aligned like on axis like we are all taught early but the elements jumble around without order. 3) Definition: A subset A of an ordered field F is bounded from below iff there exists p in F such that (for all x in A) (p <= x) Question: 1) Why as convention "bounded set" is used instead of "bounded field"? 2) Could a set being "not ordered" and "bounded" at the same time? 3) Regarding optimization view point, is polytope view of a bounded convex set totally correct? Is polytope view just to help imagining? I.e. Polytope view implied ordered-ness on a set by implying spatial location of set element. A set is simp" Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images."""
]
enc = tiktoken.get_encoding("gpt2")
MODELS = [
    (gp2_model_name, GPT(GPTConfig())),
    (conv_model_name, ConvAttention(ConvConfig())),
    (prog_model_name, ProgressiveDiverging(ProgConfig())),
]
response = {}
space_token = enc.encode(" ")
max_length = 2048
device = "cuda"
device_type = "cuda"

for model_name, model in MODELS:
    checkpoint_path = BaseTrainer.get_checkpoint_path(model_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"Loaded {model_name}")
        response[model_name] = []
        for prompt in TEST_PROMPTS:
            start_time = time.time()
            response[model_name].append(
                model.generate_text(prompt, max_length, device, device_type)
            )
            print(f"elapsed time: {(time.time() - start_time) * 1000 :.2f} ms")

for idx, prompt in enumerate(TEST_PROMPTS):
    print(f"Prompt: {prompt}")
    for model_name in response.keys():
        print(f"    {model_name}: \n{response[model_name][idx]}\n")
