"""
知识库服务模块

管理 verb_noun.json 和 pre_noun.json 的读写操作。
"""

import json
import os
from typing import List, Dict


class KnowledgeService:
    """
    知识库服务
    
    职责：
    1. 加载和管理名词-动词映射（verb_noun.json）
    2. 加载和管理名词先验描述（pre_noun.json）
    3. 批量更新知识库
    4. 查询已知名词
    
    使用示例：
        service = KnowledgeService()
        known_nouns = service.get_known_nouns()
        service.update_verb_noun([{"cup": ["hold", "pour"]}])
    """
    
    def __init__(
        self,
        verb_noun_path: str = 'context/verb_noun.json',
        pre_noun_path: str = 'context/pre_noun.json'
    ):
        """
        初始化知识库服务
        
        Args:
            verb_noun_path: verb_noun.json 文件路径
            pre_noun_path: pre_noun.json 文件路径
        """
        self.verb_noun_path = verb_noun_path
        self.pre_noun_path = pre_noun_path
        
        # 缓存数据
        self._verb_noun_cache = None
        self._pre_noun_cache = None
    
    def load_verb_noun(self) -> Dict[str, List[str]]:
        """
        加载名词-动词映射
        
        Returns:
            Dict[str, List[str]]: {noun: [verb1, verb2, ...]}
        """
        if self._verb_noun_cache is not None:
            return self._verb_noun_cache
        
        try:
            with open(self.verb_noun_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                print(f"[警告] {self.verb_noun_path} 不是 dict 格式，重新初始化")
                data = {}
            
            self._verb_noun_cache = data
            return data
            
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"[警告] {self.verb_noun_path} 不存在或格式错误，返回空字典")
            self._verb_noun_cache = {}
            return {}
    
    def load_pre_noun(self) -> List[Dict]:
        """
        加载名词先验描述
        
        Returns:
            List[Dict]: [{"key": noun, "generated_text": description, "frequency": count}, ...]
        """
        if self._pre_noun_cache is not None:
            return self._pre_noun_cache
        
        try:
            with open(self.pre_noun_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"[警告] {self.pre_noun_path} 不是 list 格式，重新初始化")
                data = []
            
            self._pre_noun_cache = data
            return data
            
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"[警告] {self.pre_noun_path} 不存在或格式错误，返回空列表")
            self._pre_noun_cache = []
            return []
    
    def get_known_nouns(self) -> set:
        """
        获取所有已知名词
        
        Returns:
            set: 已知名词集合
        """
        verb_noun = self.load_verb_noun()
        return set(verb_noun.keys())
    
    def is_noun_known(self, noun: str) -> bool:
        """
        检查名词是否已知
        
        Args:
            noun: 名词
            
        Returns:
            bool: 是否已知
        """
        known_nouns = self.get_known_nouns()
        return noun in known_nouns
    
    def update_verb_noun(self, new_entries: List[Dict[str, List[str]]]) -> None:
        """
        批量更新名词-动词映射
        
        Args:
            new_entries: 新条目列表，每个元素是 {noun: [verb1, verb2, ...]}
        """
        # 加载现有数据
        verb_noun = self.load_verb_noun()
        
        # 更新或添加
        for entry in new_entries:
            verb_noun.update(entry)
        
        # 保存
        with open(self.verb_noun_path, 'w', encoding='utf-8') as f:
            json.dump(verb_noun, f, ensure_ascii=False, indent=2)
        
        # 清除缓存
        self._verb_noun_cache = verb_noun
        
        print(f"[成功] 已更新 {len(new_entries)} 个名词到 {self.verb_noun_path}")
    
    def update_pre_noun(self, new_entries: List[Dict]) -> None:
        """
        批量更新名词先验描述
        
        Args:
            new_entries: 新条目列表，每个元素是 {"key": noun, "generated_text": desc, "frequency": count}
        """
        # 加载现有数据
        pre_noun = self.load_pre_noun()
        
        # 构建快速查找字典
        key_to_index = {item["key"]: idx for idx, item in enumerate(pre_noun)}
        
        # 更新或添加
        for entry in new_entries:
            key = entry["key"]
            if key in key_to_index:
                # 更新现有条目
                idx = key_to_index[key]
                pre_noun[idx] = entry
            else:
                # 添加新条目
                pre_noun.append(entry)
        
        # 保存
        with open(self.pre_noun_path, 'w', encoding='utf-8') as f:
            json.dump(pre_noun, f, ensure_ascii=False, indent=2)
        
        # 清除缓存
        self._pre_noun_cache = pre_noun
        
        print(f"[成功] 已更新 {len(new_entries)} 个名词到 {self.pre_noun_path}")
    
    def get_noun_description(self, noun: str) -> str:
        """
        获取名词的描述
        
        Args:
            noun: 名词
            
        Returns:
            str: 名词描述，未知返回 None
        """
        pre_noun = self.load_pre_noun()
        
        for item in pre_noun:
            if item["key"] == noun:
                return item.get("generated_text")
        
        return None
    
    def clear_cache(self) -> None:
        """清除缓存，强制重新加载"""
        self._verb_noun_cache = None
        self._pre_noun_cache = None
