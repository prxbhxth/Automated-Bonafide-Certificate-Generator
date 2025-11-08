from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
from datetime import datetime
import io
import logging
import os
from pathlib import Path
import traceback

from src.hybrid_extractor import HybridExtractor
from src.pdf_generator import generate_bonafide_pdf

app = Flask(__name__)
CORS(app)

print("Initializing extractor...")
extractor = HybridExtractor(use_bert=True)
print(f"Using extraction method: {extractor.get_extraction_method()}")

@app.route('/')
def root():
    return redirect(url_for('generate_page'))

@app.route('/generate')
def generate_page():
    return render_template('index.html')

@app.route('/api/extract', methods=['POST'])
def extract_entities():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        text = data['text']
        extracted = extractor.extract_all(text)
        is_valid, missing = extractor.validate_extraction(extracted)
        return jsonify({
            'success': True,
            'extracted': extracted,
            'is_valid': is_valid,
            'missing_fields': missing,
            'method': extractor.get_extraction_method()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route("/details")
def details_page():
    model_info = {
        "Extraction Method": "Hybrid BERT + Rule-based Entity Extraction",
        "Model 1": "bert-base-cased (fine-tuned for Named Entity Recognition)",
        "Model 2": "bert-base-uncased (trained for purpose classification)",
        "Framework": "HuggingFace Transformers + Flask",
        "PDF Engine": "ReportLab",
        "Version": "1.0.0"
    }
    return render_template("details.html", info=model_info, active_page="details")

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        required = ['name', 'roll_number', 'course', 'year', 'purpose']
        missing = [f for f in required if f not in data or not data[f]]
        if missing:
            return jsonify({'success': False, 'error': f'Missing fields: {", ".join(missing)}'}), 400
        data['date'] = datetime.now().strftime('%B %d, %Y')
        pdf_buffer = generate_bonafide_pdf(data)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'bonafide_{data["roll_number"]}.pdf'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Bonafide Certificate Generator")
    print("="*60)
    print(f"Server: http://localhost:5000")
    print(f"Method: {extractor.get_extraction_method()}")
    print("\nAPI Endpoints:")
    print("   GET  /generate            - Generator UI")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)