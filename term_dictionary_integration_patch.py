
# 在prompt_template_system.py中添加以下方法到TermDictionary类

def add_term(self, term: str, explanation: str):
    """添加术语"""
    if not term or not explanation:
        raise ValueError("术语和解释不能为空")
    self.terms[term] = explanation

def update_term(self, term: str, new_explanation: str):
    """修改术语解释"""
    if term not in self.terms:
        raise ValueError(f"术语 '{term}' 不存在")
    self.terms[term] = new_explanation

def delete_term(self, term: str):
    """删除术语"""
    if term not in self.terms:
        raise ValueError(f"术语 '{term}' 不存在")
    return self.terms.pop(term)

def search_terms(self, keyword: str):
    """搜索术语"""
    keyword_lower = keyword.lower()
    results = {}
    for term, explanation in self.terms.items():
        if (keyword_lower in term.lower() or 
            keyword_lower in explanation.lower()):
            results[term] = explanation
    return results

def save_to_csv(self, csv_path: str):
    """保存术语到CSV文件"""
    import os
    import csv
    
    # 确保目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['term', 'explanation'])
        for term, explanation in self.terms.items():
            writer.writerow([term, explanation])
