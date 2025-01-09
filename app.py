from flask import Flask, request, jsonify, render_template
import nltk
from typing import Dict, List
import re
import json

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

class SQLGenerator:
    def __init__(self):
        self.init_patterns()

    def init_patterns(self):
        # Enhanced patterns for operations
        self.operations = {
            'select': r'\b(find|get|show|display|list|select|count|retrieve|what|who|which|give me|fetch|search|group)\b',
            'insert': r'\b(add|insert|create|new|put|register|include|store)\b',
            'update': r'\b(update|modify|change|set|edit|alter|revise)\b',
            'delete': r'\b(delete|remove|drop|eliminate|erase)\b',
            'join': r'\b(join|combine|merge|connect|link)\b',
            'order': r'\b(order|sort|arrange)\b',
            'having': r'\b(having|with condition|where grouped)\b',
            'distinct': r'\b(unique|distinct|different)\b'
        }
        
        # Enhanced patterns for conditions and functions
        self.conditions = {
            'equals': r'\b(is|equals|=|same as|matching|equal to)\b',
            'greater': r'\b(greater|more|higher|>|above|over|exceeds)\b',
            'less': r'\b(less|lower|below|<|under|beneath)\b',
            'like': r'\b(like|contains|similar to|matches|starts with|ends with)\b',
            'in': r'\b(in|within|any of|among|included in)\b',
            'between': r'\b(between|from .* to|range|ranging)\b',
            'not': r'\b(not|isn\'t|aren\'t|doesn\'t|don\'t|excluding)\b',
            'null': r'\b(null|empty|missing|undefined)\b',
            'and': r'\b(and|also|both|along with)\b',
            'or': r'\b(or|either|any of)\b'
        }

        # Aggregate function patterns
        self.aggregates = {
            'count': r'\b(count|how many|number of)\b',
            'sum': r'\b(sum|total|add up)\b',
            'average': r'\b(average|avg|mean)\b',
            'maximum': r'\b(maximum|max|highest|top)\b',
            'minimum': r'\b(minimum|min|lowest|bottom)\b'
        }

        # Join type patterns
        self.join_types = {
            'inner': r'\b(inner|matching|both)\b',
            'left': r'\b(left|keeping all from first|all from left)\b',
            'right': r'\b(right|keeping all from second|all from right)\b',
            'full': r'\b(full|all|complete|outer)\b'
        }

        # Additional operation patterns
        self.additional_ops = {
            'limit': r'\b(limit|only|just|top|first)\b',
            'offset': r'\b(offset|skip|start from)\b',
            'case': r'\b(case|when|then|else|condition)\b',
            'exists': r'\b(exists|present|available)\b',
            'union': r'\b(union|combine|merge)\b',
            'intersect': r'\b(intersect|common|shared)\b',
            'except': r'\b(except|exclude|minus)\b'
        }

        # Add compound condition patterns
        self.compound_patterns = {
            'age_and_grade': r'(\d+)\s*(?:years?\s*old|\s*age)\s*(?:and|&)\s*(?:in)?\s*grade\s*(\d+)',
            'grade_and_age': r'grade\s*(\d+)\s*(?:and|&)\s*(\d+)\s*(?:years?\s*old|\s*age)',
            'multiple_conditions': r'(?:with|having|where)?\s*(\w+)\s*(=|>|<|\bis\b|\bequals?\b)\s*(\d+|\w+)\s*(?:and|&)\s*(\w+)\s*(=|>|<|\bis\b|\bequals?\b)\s*(\d+|\w+)'
        }

    def generate_sql(self, query: str, schema: Dict) -> str:
        try:
            tokens = nltk.word_tokenize(query.lower())
            pos_tags = nltk.pos_tag(tokens)
            
            operation = self.detect_operation(query.lower())
            if not operation:
                return "Error: Could not detect a valid SQL operation."

            entities = self.extract_entities(tokens, pos_tags, schema)
            conditions = self.extract_conditions(tokens, pos_tags, schema)

            # Handle sorting
            sort_column = next((cond['value'] for cond in conditions if cond['column'] == 'sort'), None)
            sort_order = next((cond['operator'] for cond in conditions if cond['column'] == 'sort'), None)

            # Handle grouping
            group_column = next((cond['value'] for cond in conditions if cond['column'] == 'group'), None)

            # Handle complex queries
            if self.is_complex_query(entities):
                return self.build_complex_query(entities, conditions, schema)
            
            # Handle basic operations
            if operation == 'select':
                sql = self.build_select(entities, conditions, schema)
                if sort_column:
                    sql += f" ORDER BY {sort_column} {sort_order}"
                if group_column:
                    sql += f" GROUP BY {group_column}"
                return sql
            elif operation == 'insert':
                return self.build_insert(entities, schema)
            elif operation == 'update':
                return self.build_update(entities, conditions, schema)
            elif operation == 'delete':
                return self.build_delete(entities, conditions, schema)
            else:
                return f"Error: Operation '{operation}' not fully implemented yet."
        except Exception as e:
            return f"Error: {str(e)}"

    def detect_operation(self, text: str) -> str:
        """Detect the main SQL operation from the text"""
        # Always treat grouping queries as SELECT operations
        if any(word in text.lower() for word in ['group', 'grouped', 'groups']):
            return 'select'
        
        for op, pattern in self.operations.items():
            if re.search(pattern, text):
                return op
        
        # Default to select if no specific operation is detected
        return 'select'

    def extract_entities(self, tokens, pos_tags, schema: Dict) -> Dict:
        entities = {
            'tables': set(['students']),  # Default table
            'columns': set(),
            'values': [],
            'query_text': ' '.join(tokens)
        }
        
        text = ' '.join(tokens).lower()
        
        # Extract columns based on the query content
        if 'grade' in text or any(str(i) for i in range(1, 13) if f'grade {i}' in text or f'grade{i}' in text):
            entities['columns'].add('grade')
        if any(word in text for word in ['age', 'old', 'year', 'years']):
            entities['columns'].add('age')
        if any(word in text for word in ['name', 'called', 'named']):
            entities['columns'].add('name')
        if 'school' in text:
            entities['columns'].add('school')
        
        # Extract values
        for token, pos in pos_tags:
            if token.isdigit():
                entities['values'].append(token)
            elif pos in ['NNP', 'NNPS'] and token.lower() not in ['students', 'student', 'show', 'find', 'get']:
                entities['values'].append(token)
        
        return entities

    def extract_conditions(self, tokens, pos_tags, schema: Dict) -> List[Dict]:
        conditions = []
        text = ' '.join(tokens).lower()
        
        # Handle grouping with improved patterns
        group_patterns = [
            r'group(?:ed)?\s+by\s+(?:their\s+)?(\w+)',
            r'group(?:ed)?\s+by\s+(?:the\s+)?(\w+)',
            r'by\s+(?:their\s+)?(\w+)s?\s+group',
            r'in\s+each\s+(\w+)',
            r'group\s+students\s+by\s+(\w+)',
            r'group\s+by\s+(\w+)'
        ]
        
        # Only try to match group patterns if we haven't found one yet
        for pattern in group_patterns:
            group_match = re.search(pattern, text)
            if group_match:
                group_column = group_match.group(1).rstrip('s')  # Remove trailing 's' if present
                if group_column in ['school', 'grade', 'age', 'name']:  # Validate column
                    conditions.append({
                        'type': 'group',  # Add type to distinguish from other conditions
                        'column': group_column,
                        'operator': 'GROUP BY',
                        'value': group_column
                    })
                    break  # Found a valid group, stop checking patterns

        # Handle average age condition
        avg_patterns = [
            r'average\s+age\s+of\s+students\s+in\s+each\s+(\w+)',
            r'average\s+age\s+by\s+(\w+)',
            r'average\s+age\s+group(?:ed)?\s+by\s+(\w+)'
        ]
        
        for pattern in avg_patterns:
            avg_match = re.search(pattern, text)
            if avg_match:
                group_by = avg_match.group(1).rstrip('s')
                if group_by in ['school', 'grade']:
                    conditions.append({
                        'column': 'average_age',
                        'operator': 'AVG',
                        'value': 'age'
                    })
                    if not any(c.get('type') == 'group' for c in conditions):
                        conditions.append({
                            'type': 'group',
                            'column': group_by,
                            'operator': 'GROUP BY',
                            'value': group_by
                        })
                    break

        # Handle "not in" conditions
        not_in_pattern = r'not\s+in\s+["\']?([^"\']+)["\']?'
        not_in_match = re.search(not_in_pattern, text)
        if not_in_match:
            value = not_in_match.group(1)
            conditions.append({
                'column': 'school',
                'operator': 'NOT IN',
                'value': f"('{value}')"
            })

        # Handle sorting
        sort_pattern = r'sorted\s+by\s+(\w+)\s+(ascending|descending)'
        sort_match = re.search(sort_pattern, text)
        if sort_match:
            sort_column = sort_match.group(1)
            sort_order = 'ASC' if sort_match.group(2) == 'ascending' else 'DESC'
            conditions.append({
                'column': 'sort',
                'operator': sort_order,
                'value': sort_column
            })

        # Age conditions with OR support
        age_or_pattern = r'(?:age\s+)?(?:is\s+)?(\d+)(?:\s*(?:or|and)\s*(\d+))?\s*(?:years?\s*old)?'
        age_match = re.search(age_or_pattern, text)
        if age_match:
            ages = [age for age in age_match.groups() if age]
            if len(ages) > 1:
                conditions.append({
                    'column': 'age',
                    'operator': 'IN',
                    'value': f"({', '.join(ages)})"
                })
            else:
                conditions.append({
                    'column': 'age',
                    'operator': '=',
                    'value': ages[0]
                })

        return conditions

    def get_columns(self, schema: Dict) -> List[str]:
        """Helper method to get all columns from schema"""
        columns = []
        for table_info in schema.values():
            columns.extend(table_info.get('columns', []))
        return columns

    def map_condition_to_operator(self, condition_type: str) -> str:
        operator_map = {
            'equals': '=',
            'greater': '>',
            'less': '<',
            'like': 'LIKE',
            'in': 'IN',
            'between': 'BETWEEN',
            'not': '!=',
            'greater_equals': '>=',
            'less_equals': '<='
        }
        return operator_map.get(condition_type, '=')

    def guess_column(self, value_type: str, schema: Dict) -> str:
        # Guess a column name based on value type (e.g., numeric or text)
        for table, details in schema.items():
            for column in details.get("columns", []):
                if value_type == 'number' and 'grade' in column.lower():
                    return column
        return 'unknown_column'

    def build_select(self, entities, conditions, schema):
        # Get the actual table name from schema
        table = next(iter(entities['tables']), list(schema.keys())[0] if schema else "unknown_table")

        # Determine columns to select
        columns = []

        # Check for average age condition
        if any(cond['column'] == 'average_age' for cond in conditions):
            columns.append("AVG(age) AS average_age")
            # If we're calculating average, we should also include the grouping column
            group_cond = next((cond for cond in conditions if cond.get('type') == 'group'), None)
            if group_cond:
                columns.append(group_cond['column'])

        # If no specific columns are requested, select all
        if not columns:
            columns.append("*")

        # Build WHERE clause with proper AND conditions
        where_clauses = []
        group_by_clause = None

        for cond in conditions:
            # Handle grouping separately
            if cond.get('type') == 'group':
                group_by_clause = f" GROUP BY {cond['column']}"
                continue

            # Skip non-where conditions
            if cond['column'] in ['sort', 'group', 'average_age']:
                continue

            column = cond['column']
            operator = cond['operator']
            value = cond['value']

            if operator == 'LIKE':
                where_clauses.append(f"{column} {operator} '{value}'")
            elif column in ['name', 'school']:  # Text columns
                where_clauses.append(f"{column} {operator} '{value}'")
            else:  # Numeric columns (grade, age)
                where_clauses.append(f"{column} {operator} {value}")

        # Construct the query
        query = f"SELECT {', '.join(columns)} FROM {table}"
        
        # Add WHERE clause if needed
        if where_clauses:
            query += f" WHERE {' AND '.join(where_clauses)}"
        
        # Add GROUP BY if it exists
        if group_by_clause:
            query += group_by_clause

        return query

    def build_insert(self, entities, schema):
        # Get the actual table name from schema
        table = next(iter(entities['tables']), list(schema.keys())[0] if schema else "unknown_table")
        
        # Get columns from schema if not specified
        if not entities['columns']:
            entities['columns'] = schema[table]['columns']
        
        # Handle multiple value sets
        value_sets = []
        values = entities['values']
        
        # If we have multiple sets of values
        if len(values) > len(entities['columns']):
            for i in range(0, len(values), len(entities['columns'])):
                value_set = values[i:i + len(entities['columns'])]
                value_sets.append(value_set)
        else:
            value_sets = [values]
        
        # Format values
        formatted_values = []
        for value_set in value_sets:
            formatted_set = []
            for val in value_set:
                if val.lower() == 'null':
                    formatted_set.append('NULL')
                elif val.isdigit():
                    formatted_set.append(val)
                else:
                    formatted_set.append(f"'{val}'")
            formatted_values.append(f"({', '.join(formatted_set)})")
        
        columns = f"({', '.join(entities['columns'])})"
        values = ', '.join(formatted_values)
        
        return f"INSERT INTO {table} {columns} VALUES {values}"

    def build_update(self, entities, conditions, schema):
        # Get the actual table name from schema
        table = next(iter(entities['tables']), list(schema.keys())[0] if schema else "unknown_table")
        
        # Build SET clause with type checking
        set_clauses = []
        for col, val in zip(entities['columns'], entities['values']):
            # Check if the value should be quoted
            if val.lower() == 'null':
                formatted_val = 'NULL'
            elif val.isdigit():
                formatted_val = val
            else:
                formatted_val = f"'{val}'"
            
            # Handle special cases like increment/decrement
            if any(op in entities.get('query_text', '').lower() for op in ['increase', 'increment', 'add to']):
                set_clauses.append(f"{col} = {col} + {formatted_val}")
            elif any(op in entities.get('query_text', '').lower() for op in ['decrease', 'decrement', 'subtract from']):
                set_clauses.append(f"{col} = {col} - {formatted_val}")
            else:
                set_clauses.append(f"{col} = {formatted_val}")
        
        set_clause = ", ".join(set_clauses)
        
        # Build WHERE clause
        where_clauses = []
        for cond in conditions:
            column = cond['column']
            operator = cond['operator']
            value = f"'{cond['value']}'" if not str(cond['value']).isdigit() else cond['value']
            where_clauses.append(f"{column} {operator} {value}")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        return f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

    def build_delete(self, entities, conditions, schema):
        # Get the actual table name from schema
        table = next(iter(entities['tables']), list(schema.keys())[0] if schema else "unknown_table")
        
        # Build WHERE clause
        where_clauses = []
        for cond in conditions:
            column = cond['column']
            operator = cond['operator']
            value = f"'{cond['value']}'" if not str(cond['value']).isdigit() else cond['value']
            where_clauses.append(f"{column} {operator} {value}")
        
        # Handle special cases
        if 'all' in entities.get('query_text', '').lower():
            return f"DELETE FROM {table}"
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Handle LIMIT if specified
        limit_clause = ""
        query_tokens = nltk.word_tokenize(entities.get('query_text', '').lower())
        if any(word in ['first', 'top', 'limit'] for word in query_tokens):
            for token in query_tokens:
                if token.isdigit():
                    limit_clause = f" LIMIT {token}"
                    break
        
        return f"DELETE FROM {table} WHERE {where_clause}{limit_clause}"

    def build_complex_query(self, entities, conditions, schema):
        """Build complex SQL queries with joins, subqueries, and advanced operations"""
        query_parts = {
            'select': [],
            'from': [],
            'joins': [],
            'where': [],
            'group_by': [],
            'having': [],
            'order_by': [],
            'limit': None,
            'offset': None
        }

        # Handle SELECT clause with aggregates and DISTINCT
        if 'distinct' in entities.get('query_text', '').lower():
            query_parts['select'].append('DISTINCT')

        # Handle aggregates and normal columns
        columns = self.process_columns(entities, schema)
        query_parts['select'].extend(columns)

        # Handle FROM clause and JOINs
        tables = self.process_tables(entities, schema)
        query_parts['from'].extend(tables)
        
        # Handle JOINs if multiple tables are involved
        if len(tables) > 1:
            joins = self.build_joins(entities, schema)
            query_parts['joins'].extend(joins)

        # Handle WHERE conditions
        where_clauses = self.build_where_conditions(conditions, entities)
        query_parts['where'].extend(where_clauses)

        # Handle GROUP BY and HAVING
        if self.needs_grouping(entities):
            group_cols = self.build_group_by(entities)
            query_parts['group_by'].extend(group_cols)
            having = self.build_having(entities, conditions)
            if having:
                query_parts['having'].append(having)

        # Handle ORDER BY
        if self.needs_ordering(entities):
            order = self.build_order_by(entities)
            query_parts['order_by'].extend(order)

        # Handle LIMIT and OFFSET
        limit, offset = self.process_limit_offset(entities)
        if limit:
            query_parts['limit'] = limit
        if offset:
            query_parts['offset'] = offset

        # Combine all parts into final query
        return self.combine_query_parts(query_parts)

    def process_columns(self, entities, schema):
        """Process columns including aggregates and expressions"""
        columns = []
        query_text = entities.get('query_text', '').lower()
        
        # Handle aggregates
        for agg_type, pattern in self.aggregates.items():
            if re.search(pattern, query_text):
                for col in entities['columns']:
                    columns.append(f"{agg_type.upper()}({col})")
        
        # Handle regular columns
        for col in entities['columns']:
            if not any(agg in col for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                columns.append(col)
        
        # Handle CASE expressions
        if 'case' in query_text or 'when' in query_text:
            case_expr = self.build_case_expression(entities)
            if case_expr:
                columns.append(case_expr)
        
        return columns if columns else ['*']

    def build_joins(self, entities, schema):
        """Build JOIN clauses based on natural language description"""
        joins = []
        query_text = entities.get('query_text', '').lower()
        
        # Detect join type
        join_type = 'INNER JOIN'
        for jtype, pattern in self.join_types.items():
            if re.search(pattern, query_text):
                join_type = f"{jtype.upper()} JOIN"
                break
        
        # Build join conditions based on foreign key relationships or common columns
        tables = list(entities['tables'])
        for i in range(1, len(tables)):
            table1, table2 = tables[i-1], tables[i]
            common_cols = self.find_common_columns(table1, table2, schema)
            if common_cols:
                joins.append(f"{join_type} {table2} ON {table1}.{common_cols[0]} = {table2}.{common_cols[0]}")
        
        return joins

    def build_case_expression(self, entities):
        """Build CASE expression from natural language"""
        query_text = entities.get('query_text', '').lower()
        case_parts = re.findall(r'when\s+(.+?)\s+then\s+(.+?)(?=\s+when|\s+else|\s+end|$)', query_text)
        
        if case_parts:
            case_expr = "CASE "
            for condition, result in case_parts:
                case_expr += f"WHEN {condition} THEN {result} "
            if 'else' in query_text:
                else_part = re.findall(r'else\s+(.+?)(?=\s+end|$)', query_text)
                if else_part:
                    case_expr += f"ELSE {else_part[0]} "
            case_expr += "END"
            return case_expr
        return None

    def is_complex_query(self, entities):
        """Determine if the query needs complex processing"""
        query_text = entities.get('query_text', '').lower()
        complex_indicators = [
            'join', 'group by', 'having', 'case', 'when',
            'union', 'intersect', 'except', 'exists',
            'distinct', 'limit', 'offset'
        ]
        return any(indicator in query_text for indicator in complex_indicators)

    def map_operator(self, op: str) -> str:
        """Map natural language operators to SQL operators"""
        op = op.lower().strip()
        if op in ['is', 'equals', '=']:
            return '='
        elif op in ['>', 'greater', 'above', 'over']:
            return '>'
        elif op in ['<', 'less', 'below', 'under']:
            return '<'
        return '='

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-sql", methods=["POST"])
def generate_sql():
    data = request.json
    query = data.get("query", "")
    schema = data.get("schema", {})
    table_data = data.get("tableData", {})  # Get the table data
    
    try:
        generator = SQLGenerator()
        sql = generator.generate_sql(query, schema)
        return jsonify({"sql": sql})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
