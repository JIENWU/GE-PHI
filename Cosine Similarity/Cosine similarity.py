import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def calculate_cosine_similarity(seq1, seq2):
    # 使用CountVectorizer将序列转换为特征向量
    vectorizer = CountVectorizer(analyzer='char')
    vectors = vectorizer.fit_transform([seq1, seq2])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectors)[0, 1]
    return cosine_sim


def process_and_save_similarity(input_csv, output_csv):
    # 读取输入CSV文件
    df = pd.read_csv(input_csv)

    # 创建一个空的列表来保存相似度结果
    results = []

    # 遍历每一行，计算两列之间的余弦相似度
    for index, row in df.iterrows():
        seq1 = row[0]
        seq2 = row[1]
        similarity = calculate_cosine_similarity(seq1, seq2)
        results.append([seq1, seq2, similarity])

    # 将结果保存为DataFrame，并输出为CSV文件
    result_df = pd.DataFrame(results, columns=['Sequence1', 'Sequence2', 'Cosine Similarity'])
    result_df.to_csv(output_csv, index=False)
    print(f"Cosine similarities saved to {output_csv}")


if __name__ == "__main__":
    input_csv = 'host interactio network.csv'  # 输入的CSV文件路径
    output_csv = 'host cosine_similarity 所有的结果.csv'  # 输出的CSV文件路径
    process_and_save_similarity(input_csv, output_csv)
