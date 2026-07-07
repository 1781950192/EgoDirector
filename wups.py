# ============== 直接复制粘贴就行 ==============
# pip install nltk
# python 里跑一次：import nltk; nltk.download('wordnet')

import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet', quiet=True)


def wups(words_a, words_b, threshold=0.1):
    """
    输入两个短语（字符串），返回论文里用的 Activity WUPS
    例子：
        wups("take cup", "grasp mug")    → 0.706
        wups("cut tomato", "slice tomato") → 0.950
    """
    if isinstance(words_a, str):
        words_a = words_a.lower().strip().split()
    if isinstance(words_b, str):
        words_b = words_b.lower().strip().split()

    # 把动作和物体分开（论文就是这么拆的）
    action_a = words_a[0] if len(words_a) >= 1 else ""
    object_a = words_a[-1] if len(words_a) >= 2 else words_a[0] if words_a else ""
    action_b = words_b[0] if len(words_b) >= 1 else ""
    object_b = words_b[-1] if len(words_b) >= 2 else words_b[0] if words_b else ""

    # 单个词的 WUPS
    def single_wups(w1, w2):
        s1 = wn.synsets(w1, pos=wn.VERB if w1 in action_a or w1 in action_b else wn.NOUN)
        s2 = wn.synsets(w2, pos=wn.VERB if w2 in action_b or w2 in action_a else wn.NOUN)
        if not s1 or not s2:
            return 0.0
        # 取第一个 synset（论文默认做法）
        score = s1[0].wup_similarity(s2[0]) or 0.0
        return score

    action_score = single_wups(action_a, action_b)
    object_score = single_wups(object_a, object_b)

    activity_wups = action_score * object_score

    # 论文里阈值截断 0.1
    return 0.0 if activity_wups < threshold else activity_wups


