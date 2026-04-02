import torch
from sklearn.cluster import  MiniBatchKMeans

def get_frequent_tokens(dataset, tokenizer, top_n=5000, max_len=1024):
    from collections import Counter

    counter = Counter()

    for ex in dataset:
        # --- required fields ---
        if "prompt" not in ex or "solution" not in ex:
            continue

        text = ex["prompt"][1]['content'] + "\n" + ex["solution"]

        ids = tokenizer(
            text,
            add_special_tokens=False,  # 🚨 no BOS/EOS/system
            truncation=True,
            max_length=max_len,
        ).input_ids

        counter.update(ids)

    return [tid for tid, _ in counter.most_common(top_n)]

# --------------------------------------------------
# 1. Static token embedding extraction (NO forward)
# --------------------------------------------------

@torch.no_grad()
def get_token_embeddings(
    model,
    token_ids,
    device,
):
    """
    Extract static token embeddings from the model.
    Safe for PEFT, Liger, DeepSpeed.

    Returns:
        Tensor [N, H]
    """
    # unwrap PEFT
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    # HuggingFace standard
    embedding_layer = base.get_input_embeddings()
    weight = embedding_layer.weight  # [V, H]

    if device is None:
        device = weight.device

    token_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    )

    emb = weight[token_ids]  # [N, H]

    # ---- DEBUG ----
    assert emb.ndim == 2
    assert emb.shape[0] == len(token_ids)
    # ---------------

    return emb


# --------------------------------------------------
# 2. KMeans clustering → focal sets
# --------------------------------------------------

def build_kmeans_focal_sets(
    token_embeddings,
    token_ids,
    k: int,
    seed: int
):
    X = token_embeddings.float().cpu().numpy()

    if X.shape[0] < k:
        return []

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=1024,
        n_init="auto",
        max_iter=50,
        random_state=seed,
    )

    labels = kmeans.fit_predict(X)

    clusters = {}
    for tid, lab in zip(token_ids, labels):
        clusters.setdefault(lab, []).append(tid)

    return [c for c in clusters.values() if len(c) > 1]

# --------------------------------------------------
# 3. Budgeting = singleton + clustered focal sets
# --------------------------------------------------

def budget_focal_sets_rsllm(
    model,
    candidate_token_ids,
    k: int,
    device,
    seed: int = 42,
):
    """
    RS-LLM budgeting using KMeans over all candidate tokens.

    Returns:
        focal_sets
    """
    assert len(candidate_token_ids) > 0

    # 1) singleton focal sets (mandatory)
    singleton_sets = [[tid] for tid in candidate_token_ids]

    # 2) static embeddings
    token_embs = get_token_embeddings(
        model,
        candidate_token_ids,
        device=device,
    )

    print(
        f"[Budgeting] tokens={len(candidate_token_ids)} "
        f"embed_dim={token_embs.shape[1]}"
    )

    # 3) KMeans clustering
    clustered_sets = build_kmeans_focal_sets(
        token_embeddings=token_embs,
        token_ids=candidate_token_ids,
        k=k,        
        seed=seed
    )

    # 4) combine
    focal_sets = singleton_sets + clustered_sets

    print(
        f"[Budgeting] singleton={len(singleton_sets)} "
        f"clustered={len(clustered_sets)} "
        f"total={len(focal_sets)}"
    )

    return focal_sets

