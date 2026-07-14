# ~~~~~~~~~~~~~~~~~~~~
# from visual_tokens.ipynb 
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F

def visualize_tokens(prompt, tokenizer):
    ''' 
    Space (0x20) is mapped to Ġ (U+0120).
    Newlines (\n) are often mapped to Ċ (U+010A).
    "Life is good" -> ["Life", "Ġis", "Ġgood"]
    '''
    tokens = tokenizer.tokenize(prompt)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Print a "color-blocked" version to see boundaries
    print("Tokenized Output:")
    for i, (token, token_id) in enumerate(zip(tokens, ids)):
        # Alternate background colors to show where tokens start/end
        # color = "\033[43m" if i % 2 == 0 else "\033[46m" # simple color
        color = "\033[48;2;255;186;205;m" if i % 2 == 0 else "\033[48;2;123;200;246;m" # note: 48 for background, 38 for foreground, and r;g;b for color
        # better visualization for "_sth" (Llama-3.1, GPT-2, and RoBERTa)
        visual_token = token.replace('Ġ', '_') 
        print(f"{color}{visual_token}\033[0m", end="")
    print(f"\n\nRaw IDs: {ids}")

def softmax_token(scores, generated_tokens): 
    ''' 
    outputs.scores is a tuple of length 'max_new_tokens
    '''
    probs = []
    for i, logits in enumerate(scores):
        # Apply softmax to get probabilities for the whole vocab
        p = F.softmax(logits, dim=-1)
        # Extract the probability of the specific token the model actually picked
        token_id = generated_tokens[i]
        probs.append(p[0, token_id].item())
    return probs

def test_text_generation(model, tokenizer, prompt="Once upon a time", num_tokens=20,
                         do=(1,1)):
    ''' 
    modified from src/tokens2words/run_vocab_expansion_eval.py
    do: 1 for greedy, 1 for top p
    '''
    # Before calling the function, ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        print("> add pad token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Encode with return_tensors and include the mask
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
        
    def _generate_greedy(input_ids):
        # Greedy decoding
        outputs = model.generate(
                    input_ids, 
                    attention_mask=attention_mask, # Pass the mask here
                    pad_token_id=tokenizer.pad_token_id, # Explicitly set this
                    max_new_tokens=num_tokens, # Use max_new_tokens for clarity
                    return_dict_in_generate=True, # Required to get scores
                    output_scores=True,           # Required to get scores
                    do_sample=False,
                    temperature=None, 
                    top_p=None
                )
        # Get the tokens generated (excluding prompt)
        generated_tokens = outputs.sequences[0][input_ids.shape[-1]:]
        
        # outputs.sequences contains the token IDs (prompt + generation)
        # outputs.scores is a tuple of length 'max_new_tokens'
        return (outputs.scores, generated_tokens) if do[0] else None
        # simple output wo/ probability
        greedy_decoded = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        logger.info(f"Greedy:\n{greedy_decoded}\nToken IDs:\n{greedy_output[0].tolist()}")

    def _generate_top_p(input_ids):
        # Top-p sampling with temperature
        outputs = model.generate(
                            input_ids, 
                            attention_mask=attention_mask, 
                            pad_token_id=tokenizer.pad_token_id, 
                            max_new_tokens=num_tokens, 
                            return_dict_in_generate=True, # Required to get scores
                            output_scores=True,           # Required to get scores
                            do_sample=True, 
                            top_p=0.9,
                            temperature=0.7
                            )
        generated_tokens = outputs.sequences[0][input_ids.shape[-1]:]
        return (outputs.scores, generated_tokens) if do[1] else None
        top_p_decoded = tokenizer.decode(top_p_output[0], skip_special_tokens=True)
        logger.info(f"Sampling:\n{top_p_decoded}\nToken IDs:\n{top_p_output[0].tolist()}")

    return _generate_greedy(input_ids), _generate_top_p(input_ids)
    # _generate_top_p(input_ids)

def gen_prob(scores, tokenizer, generated_tokens):
    soft_scores = softmax_token(scores, generated_tokens)
    gen_decoded = tokenizer.convert_ids_to_tokens(generated_tokens)
    for i, (token, prob) in enumerate(zip(gen_decoded, soft_scores)):
        prob = 255 - int(255*prob)
        color = f"\033[48;2;255;244;{prob};m"
        visual_token = token.replace('Ġ', '_') 
        visual_token = visual_token.replace('Ċ', '~') 
        print(f"{color}{visual_token}\033[0m", end="")
    # print(f"\n\nRaw IDs: {generated_tokens}")
    print("\n<token>\t<prob>")
    for i, (token, prob) in enumerate(zip(gen_decoded, soft_scores)):
        print(f"{token}\t{prob:.3f}")

# ~~~~~~~~~~~~~~~~~~~~

t2n = lambda tensor: tensor.half().detach().cpu().numpy() # why do we have to have .to(torch.float32) for gemma3.270m?
w2n = lambda tensor: tensor.to(torch.float32).detach().cpu().numpy()

'''see
https://claude.ai/chat/fb2e36c8-52f1-48a3-98c6-fcc12d077ac6
https://chatgpt.com/c/69f90657-fd88-83ea-bcda-e15f7b324e91
'''
def _make_attn_hook(token_idx: int, start_layer: int, current_layer_holder: list, model_name: str):
    """
    Returns a pre-forward hook for an attention module.

    At layers >= start_layer the hook patches the `attention_mask` keyword
    argument so that token `token_idx` can only attend to itself.

    `current_layer_holder` is a 1-element list used as a mutable counter so
    the hook knows which layer it is currently in.  Increment it externally
    after each layer.

    Works for both:
      • models that pass attention_mask as a 4-D additive bias  (shape B,H,T,T)
      • models that pass it as a 2-D boolean / 0-1 mask        (shape B,T)
    The hook converts either form to the 4-D additive bias expected by
    scaled_dot_product_attention / F.scaled_dot_product_attention.
    """
    def hook(module, args, kwargs):
        layer = current_layer_holder[0]
        if layer < start_layer:
            current_layer_holder[0] += 1
            return  # nothing to do

        # ------------------------------------------------------------------
        # Grab attention_mask from kwargs (HF passes it as kwarg in most
        # modern models).  Fall back to positional arg[1] for older models.
        # ------------------------------------------------------------------
        mask = kwargs.get("attention_mask", None)
        if mask is None and len(args) > 1:
            mask = args[1]

        # hidden_states may be in args[0] (GPT-2 style) or kwargs (Phi/Gemma style)
        if len(args) > 0:
            hidden = args[0]
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        else:
            current_layer_holder[0] += 1
            return  # can't determine shape; skip silently
        B, T, _ = hidden.shape

        # Build a fresh 4-D additive bias:  -inf everywhere for row token_idx,
        # except the diagonal (self) stays 0.
        # NEG_INF = torch.finfo(torch.float32).min # reason: some internal paths upcast to float32 before the softmax
        # NEG_INF = torch.finfo(hidden.dtype).min
        if model_name in ["phi"]:
            NEG_INF = torch.finfo(torch.float32).min 
        elif model_name in ["gem", "pyt", "Lla", "Qwe"]:
            NEG_INF = torch.finfo(hidden.dtype).min
        else:
            NEG_INF = None
            print('> Check model_name!')
        ''' 
        note:
        gemini-2 - hidden.dtype
        phi-2 - torch.float32
        '''
        
        # Start from the existing mask so we don't break causal structure for
        # OTHER tokens (they keep their original attention pattern).
        if mask is not None and mask.dim() == 4:
            new_mask = mask.clone()
        elif mask is not None and mask.dim() == 2:
            # 2-D mask: 1 = attend, 0 = ignore  →  additive bias
            causal_2d = mask[:, None, None, :].to(hidden.dtype)
            new_mask = (1.0 - causal_2d) * NEG_INF
        else:
            # No mask provided → build a standard causal mask
            causal = torch.tril(torch.ones(T, T, device=hidden.device, dtype=hidden.dtype))
            new_mask = (1.0 - causal).unsqueeze(0).unsqueeze(0) * NEG_INF
            new_mask = new_mask.expand(B, -1, -1, -1).clone()
        
        # reason: computing the mask in float32 precision where -inf is well-defined, then casting to hidden.dtype just before writing it back
        new_mask = new_mask.to(hidden.dtype) 
        # For token_idx's row: mask out all positions except token_idx itself
        new_mask[:, :, token_idx, :] = NEG_INF           # mask everything …
        new_mask[:, :, token_idx, token_idx] = 0.0       # … then unmask self

        # Write back — kwargs takes priority since that's what Phi/Gemma use
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = new_mask
        elif len(args) > 1:
            args = (args[0], new_mask) + args[2:]
        else:
            # mask wasn't passed at all originally; inject it into kwargs
            kwargs["attention_mask"] = new_mask

        current_layer_holder[0] += 1
        return args, kwargs

    return hook

def get_top_toks(result, tokenizer, k=6):
    '''
    result = model.generate()
    get token's final scores -> softmax -> top-k
    k = 6 
    '''
    scores = result.scores[0]
    p = F.softmax(scores, dim=-1).detach().cpu().numpy()[0]
    top_k_ids = np.argpartition(p, -k)[-k:]
    top_k_ids = top_k_ids[np.argsort(p[top_k_ids])][::-1]
    top_k_val = p[top_k_ids]
    top_k_tok = tokenizer.convert_ids_to_tokens(top_k_ids)
    top_k_tok = [t.replace('Ġ', '_').replace('Ċ', '~') for t in top_k_tok]
    assert hasattr(result, "hidden_states")    
    return (top_k_ids, top_k_tok, top_k_val, result.hidden_states, scores) # id, string, prob, hiddens, scores
    # note: do we need intermediate hiddens, or only the final hidden? only final for now.
    
# ---------------------------------------------------------------------------
# Main ablation function
# ---------------------------------------------------------------------------

def ablate_and_generate(
    model, model_name,
    tokenizer,
    input_ids: torch.Tensor,               # shape (1, T)
    attention_mask: torch.Tensor,           # shape (1, T)
    layer_idx: int,                         # l  – first layer to ablate
    token_idx: int,                         # t  – the token whose context is ablated
    max_new_tokens: int = 1,
    **generate_kwargs,
) -> dict:
    """
    Run `model.generate` with the local-context ablation active for (layer_idx, token_idx).

    Returns
    -------
    dict with keys:
        "input_ids"      : original prompt ids  (1, T)
        "generated_ids"  : full ids incl. new token(s)  (1, T+k)
        "new_token_id"   : the single generated token id (int)
        "new_token_str"  : decoded string
    """
    # only want until token_idx
    input_ids     = input_ids[:, :token_idx + 1]
    attention_mask = attention_mask[:, :token_idx + 1]
    
    assert input_ids.shape[0] == 1, "Only batch_size=1 supported."
    T = input_ids.shape[1]
    assert 0 <= token_idx < T,  f"token_idx {token_idx} out of range [0, {T})"

    # -----------------------------------------------------------------------
    # Identify the transformer layers.  Works for Llama / Mistral / GPT-2 /
    # Falcon style models where layers live at model.layers or model.h.
    # -----------------------------------------------------------------------
    def _get_layers(m):
        # unwrap possible accelerate / DataParallel wrapper
        base = getattr(m, "module", m)
        for attr in ("model", "transformer"):        # causal wrapper → core
            if hasattr(base, attr):
                base = getattr(base, attr)
                break
        if model_name == 'pyt':
            base = getattr(base, "gpt_neox")
        for attr in ("layers", "h", "blocks"):       # layer list
            if hasattr(base, attr):
                return getattr(base, attr)
        raise ValueError(
            "Cannot locate transformer layers. "
            "Supported attribute paths: model.layers, transformer.h, model.h, etc. "
            "Please pass `layer_list` explicitly."
        )

    layers = _get_layers(model)
    num_layers = len(layers)
    assert 0 <= layer_idx < num_layers, \
        f"layer_idx {layer_idx} out of range [0, {num_layers})"

    # -----------------------------------------------------------------------
    # Register hooks on each attention sub-module.
    # We keep a shared counter so the hook knows which layer fired.
    # -----------------------------------------------------------------------
    # Find the attention sub-module name (varies by architecture)
    def _get_attn(layer_module):
        for attr in ("self_attn", "attn", "attention", "self_attention"):
            if hasattr(layer_module, attr):
                return getattr(layer_module, attr)
        raise ValueError(
            f"Cannot find attention sub-module in layer {layer_module}. "
            "Expected one of: self_attn, attn, attention, self_attention."
        )

    current_layer_holder = [0]   # mutable counter shared across all hooks
    handles = []
    for i, layer in enumerate(layers):
        attn = _get_attn(layer)
        h = attn.register_forward_pre_hook(
            _make_attn_hook(token_idx, layer_idx, current_layer_holder, model_name),
            with_kwargs=True,
        )
        handles.append(h)

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy by default; override via kwargs
                **generate_kwargs,
            )
    finally:
        # Always remove hooks, even if generation throws
        for h in handles:
            h.remove()

    # if return scores:
    return get_top_toks(out, tokenizer) # return out

    # original, only return ids
    new_token_id = out[0, T].item()
    return {
        "input_ids":      input_ids,
        "generated_ids":  out,
        "new_token_id":   new_token_id,
        "new_token_str":  tokenizer.decode([new_token_id]),
    }


# ---------------------------------------------------------------------------
# Batch runner: sweep all (layer, token) pairs
# ---------------------------------------------------------------------------

def ablate_all(
    model, model_name,
    tokenizer,
    prompt: str,
    **generate_kwargs,
) -> dict:
    """
    For every (layer_idx, token_idx) pair, run ablate_and_generate and collect
    the single next-token prediction.

    Returns
    -------
    results : dict mapping (layer_idx, token_idx) → result dict from ablate_and_generate
    token_matrix : torch.Tensor of shape (num_layers, T) with the predicted token id
                   for each (l, t) pair.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids     = inputs["input_ids"]        # (1, T)
    attention_mask = inputs["attention_mask"]   # (1, T)
    T = input_ids.shape[1]

    base = getattr(model, "module", model)
    for attr in ("model", "transformer"):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break
    for attr in ("layers", "h", "blocks"):
        if hasattr(base, attr):
            num_layers = len(getattr(base, attr))
            break

    results = {}
    token_matrix = torch.zeros(num_layers, T, dtype=torch.long)

    total = num_layers * T
    done  = 0
    for l in range(num_layers):
        for t in range(T):
            res = ablate_and_generate(
                model, model_name, tokenizer,
                input_ids, attention_mask,
                layer_idx=l, token_idx=t,
                **generate_kwargs,
            )            
            results[(l, t)] = (res[0], res[1], res[2], np.array([t2n(i) for i in res[3][0]]), t2n(res[4]))
            token_matrix[l, t] = res[0][0] # res["new_token_id"]
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}]  layer={l}, token={t}  "
                      f"→ '{res[1][0]}'") # f"→ '{res['new_token_str']}'")

    return results, token_matrix

# ~~~~~~~~~~~~~~~~~~~~

def ablate_final(
    model, model_name,
    tokenizer,
    prompt: str,
    **generate_kwargs,
) -> dict:
    """
    For every (layer_idx, token_idx) pair, run ablate_and_generate and collect
    the single next-token prediction. Only the final token position. 

    Returns
    -------
    results : dict mapping (layer_idx, token_idx) → result dict from ablate_and_generate
    token_matrix : torch.Tensor of shape (num_layers, T) with the predicted token id
                   for each (l, t) pair.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids     = inputs["input_ids"]        # (1, T)
    attention_mask = inputs["attention_mask"]   # (1, T)
    T = input_ids.shape[1]

    base = getattr(model, "module", model)
    for attr in ("model", "transformer"):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break
    if model_name == 'pyt':
        base = getattr(base, "gpt_neox")
    for attr in ("layers", "h", "blocks"):
        if hasattr(base, attr):
            num_layers = len(getattr(base, attr))
            break

    results = {}
    token_matrix = torch.zeros(num_layers, T, dtype=torch.long)

    total = num_layers * T
    done  = 0
    for l in range(num_layers):
        # for t in range(T):
        t = T - 1
        res = ablate_and_generate(
            model, model_name, tokenizer,
            input_ids, attention_mask,
            layer_idx=l, token_idx=t,
            **generate_kwargs,
        )            
        results[(l, t)] = (res[0], res[1], res[2], np.array([t2n(i) for i in res[3][0]]), res[4])
        token_matrix[l, t] = res[0][0] # res["new_token_id"]
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  [{done}/{total}]  layer={l}, token={t}  "
                    f"→ '{res[1][0]}'") # f"→ '{res['new_token_str']}'")

    return results, token_matrix
# ~~~~~~~~~~~~~~~~~~~~
