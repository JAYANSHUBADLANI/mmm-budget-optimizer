{# Use the schema configured on the model rather than prefixing it with the target schema,
   which keeps the bronze, silver and gold names readable in DuckDB. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
