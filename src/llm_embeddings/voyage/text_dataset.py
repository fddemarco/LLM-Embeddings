import tiktoken


MAX_INPUT_TOKENS = 8191
EMBEDDING_ENCODING = "cl100k_base"


class Dataset:
    def __init__(self, data, text_key, group_key):
        self.data = data
        self.text_key = text_key
        self.group_key = group_key

    def grouped(self):
        return (
            self.data.groupby(self.group_key)[self.text_key]
            .apply(" ".join)
            .reset_index()
        )

    def texts(self):
        groups = self.grouped()
        return groups[self.text_key].to_list()

    def groups(self):
        return self.grouped()[self.group_key].tolist()


class LongDataset(Dataset):
    def texts(self):
        texts = self.__super__().texts()
        return self.truncate_texts(texts)

    def truncate_texts(self, texts):
        encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
        return [
            encoder.encode(text.replace("\n", " "))[:MAX_INPUT_TOKENS] for text in texts
        ]
