import json
import paras

class Schema:
    def __init__(self, subject_type, predicate, object_type):
        self.subject_type = subject_type
        self.predicate = predicate
        self.object_type = object_type
    
    def match(self, subject_type, object_type):
        return self.subject_type == subject_type and self.object_type == object_type

    def match2(self, subject_type, predicate, object_type):
        return self.subject_type == subject_type and self.predicate == predicate and self.object_type == object_type

    def output_result(self):
        res = {'subject_type': self.subject_type, 'predicate': self.predicate, 'object_type': self.object_type}
        return res

    @staticmethod
    def default_result():
        res = {'subject_type': '', 'predicate': '', 'object_type': ''}
        return res

class Schemas:
    def __init__(self):
        self.schemas = []
        self.types = []
        self.type_len = 0
    
    def find_matched_schema_ids(self, subject_type, object_type):
        res = []
        for i, schema in enumerate(self.schemas):
            if schema.match(subject_type, object_type):
                res.append(i)
        return res

    def predicate_to_id(self, subject_type, predicate, object_type):
        for i, schema in enumerate(self.schemas):
            if schema.match2(subject_type, predicate, object_type):
                return i
        raise ValueError

    def add_schema(self, s):
        self.schemas.append(s)
        if s.subject_type not in self.types:
            self.types.append(s.subject_type)
        if s.object_type not in self.types:
            self.types.append(s.object_type)
        self.type_len = len(self.types)

    def entity_to_id(self, entity_type):
        if entity_type in self.types:
            return self.types.index(entity_type)
        else:
            return -1

    def output_result(self, index):
        if index in range(50):
            return self.schemas[index].output_result()
        else:
            return Schema.default_result()

    def print_info(self):
        print('schema number:', len(self.schemas))
        print('entity types:', len(self.types))

def load_schema():
    s = Schemas()
    with open(paras.SCHEMAS, 'r', encoding='utf-8') as f:
        tmp = f.read().split('\n')
        for line in tmp:
            if line == '':
                continue
            a = json.loads(line.lower())
            b = Schema(a['subject_type'], a['predicate'], a['object_type'])
            s.add_schema(b)
    return s

if __name__ == '__main__':
    s = load_schema()
    s.print_info()
    for schema in s.schemas:
        print(s.find_matched_schema_ids(schema.subject_type, schema.object_type))