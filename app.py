from flask import Flask, request, jsonify, render_template
import spacy
from typing import Dict, List
import re

app = Flask(__name__)

class SQLGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.init_patterns()

    def init_patterns(self):
        # Enhanced patterns for operations
        self.operations = {
            'select': r'\b(find|get|show|display|list|select|count|retrieve|what|who|which|give me|fetch)\b',
            'insert': r'\b(add|insert|create|new)\b',
            'update': r'\b(update|modify|change|set)\b',
            'delete': r'\b(delete|remove|drop)\b'
        }
        # Patterns for conditions
        self.conditions = {
            'equals': r'\b(is|equals|=|same as|matching)\b',
            'greater': r'\b(greater|more|higher|>|above|over)\b',
            'less': r'\b(less|lower|below|<|under)\b',
            'like': r'\b(like|contains|similar to|matches)\b',
            'in': r'\b(in|within|any of)\b',
            'between': r'\b(between|from .* to|range)\b'
        }

    def generate_sql(self, query: str, schema: Dict) -> str:
        try:
            doc = self.nlp(query.lower())
            operation = self.detect_operation(doc.text)
            if not operation:
                # Fallback to SELECT if no valid operation is detected
                operation = 'select'

            entities = self.extract_entities(doc, schema)
            conditions = self.extract_conditions(doc, schema)

            if operation == 'select':
                return self.build_select(entities, conditions, schema)
            else:
                return f"Error: Operation '{operation}' not fully implemented yet."
        except Exception as e:
            return f"Error: {str(e)}"

    def detect_operation(self, text: str) -> str:
        for op, pattern in self.operations.items():
            if re.search(pattern, text):
                return op
        return None

    def extract_entities(self, doc, schema: Dict) -> Dict:
        entities = {'tables': set(), 'columns': set()}
        for token in doc:
            if token.text.lower() in schema.keys():
                entities['tables'].add(token.text.lower())
            elif token.dep_ in ['nsubj', 'dobj', 'pobj'] or token.ent_type_ in ["CARDINAL", "ORDINAL"]:
                entities['columns'].add(token.text.lower())
        return entities

    def extract_conditions(self, doc, schema: Dict) -> List[Dict]:
        conditions = []
        text = doc.text

        # Try to extract numeric and text-based conditions
        for token in doc:
            if token.like_num:  # Numeric values
                column = self.guess_column('number', schema)
                conditions.append({'column': column, 'operator': '=', 'value': token.text})
            elif token.ent_type_ == "ORDINAL":
                column = self.guess_column('number', schema)
                conditions.append({'column': column, 'operator': '=', 'value': token.text})
        return conditions

    def guess_column(self, value_type: str, schema: Dict) -> str:
        # Guess a column name based on value type (e.g., numeric or text)
        for table, details in schema.items():
            for column in details.get("columns", []):
                if value_type == 'number' and 'grade' in column.lower():
                    return column
        return 'unknown_column'

    def build_select(self, entities, conditions, schema):
        # Use the detected table or default to "students"
        table = next(iter(entities['tables']), "students")
        columns = ", ".join(entities['columns']) if entities['columns'] else "*"

        # Build the WHERE clause
        where_clauses = []
        for cond in conditions:
            column = cond['column']
            operator = cond['operator']
            value = f"'{cond['value']}'" if not value.isdigit() else cond['value']
            where_clauses.append(f"{column} {operator} {value}")

        where_clause = " AND ".join(where_clauses)
        return f"SELECT {columns} FROM {table}" + (f" WHERE {where_clause}" if where_clause else "")

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-sql", methods=["POST"])
def generate_sql():
    data = request.json
    query = data.get("query", "")
    schema = data.get("schema", {})
    generator = SQLGenerator()
    sql = generator.generate_sql(query, schema)
    return jsonify({"sql": sql})

if __name__ == "__main__":
    app.run(debug=True)
