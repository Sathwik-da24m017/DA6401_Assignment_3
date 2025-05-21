import json

# ------------------------------
# Save attention visualizations as HTML
# ------------------------------
def save_interactive_attention_html(examples, filename="attention_visualization.html", threshold=0.15):
    data_js = json.dumps(examples, ensure_ascii=False)

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Interactive Attention Visualization</title>
    <style>
      body {{ font-family: Inter, sans-serif; background: #f4f6fa; }}
      h2   {{ text-align: center; color: #3f51b5; }}
      .card {{ background: #fff; padding: 20px; margin: 16px; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); display: inline-block; }}
      .tok {{ padding: 4px 6px; margin: 2px; border-radius: 4px; background: #eee; display: inline-block; font-family: monospace; cursor: pointer; transition: background 0.15s; }}
      .inputtok {{ background: #ddd; }}
    </style>
    </head>
    <body>
    <h2>Interactive Attention Visualization</h2>
    <div id="root"></div>

    <script>
    const DATA = {data_js};
    const THRESHOLD = {threshold};

    function attColor(att) {{
      const g = Math.round(255 - 180 * att);
      return `rgb(${g},255,${g})`;
    }}

    const root = document.getElementById("root");

    Object.entries(DATA).forEach(([word, models]) => {{
      const section = document.createElement("div");
      section.innerHTML = `<h3 style='margin-top:30px;'>Input: ${word}</h3>`;

      Object.entries(models).forEach(([model, ex]) => {{
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<h4>Model: ${model.toUpperCase()}</h4>`;

        const inputRow = document.createElement("div");
        ex.input.slice(1, -1).forEach((ch, i) => {{
          const span = document.createElement("span");
          span.textContent = ch;
          span.className = "tok inputtok";
          span.id = `${word}_${model}_in_${i}`;
          inputRow.appendChild(span);
        }});

        const outputRow = document.createElement("div");
        ex.output.forEach((ch, i) => {{
          const span = document.createElement("span");
          span.textContent = ch;
          span.className = "tok";

          span.onmouseenter = () => {{
            ex.att[i]?.forEach((val, j) => {{
              const el = document.getElementById(`${word}_${model}_in_${j}`);
              if (el) el.style.background = val >= THRESHOLD ? attColor(val) : "#eee";
            }});
          }};

          span.onmouseleave = () => {{
            ex.input.slice(1, -1).forEach((_, j) => {{
              const el = document.getElementById(`${word}_${model}_in_${j}`);
              if (el) el.style.background = "#ddd";
            }});
          }};

          outputRow.appendChild(span);
        }});

        card.appendChild(inputRow);
        card.appendChild(document.createElement("br"));
        card.appendChild(outputRow);
        section.appendChild(card);
      }});

      root.appendChild(section);
    }});
    </script>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"✅ Interactive attention visual saved to: {filename}")
