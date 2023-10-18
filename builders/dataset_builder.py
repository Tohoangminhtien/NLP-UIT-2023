from .registry import Registry

META_DATASET = Registry("DATASET")

def build_dataset(annotation_path, vocab, config):
    if annotation_path is None:
        return None
    
    dataset = META_DATASET.get(config.TYPE)(annotation_path, vocab, config)

    return dataset