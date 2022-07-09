

def get_top_k_from_dict(join_paths: dict, k: int):
    return {key: join_paths[key] for i, key in enumerate(join_paths) if i < k}