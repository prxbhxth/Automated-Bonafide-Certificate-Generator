from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import io

def generate_bonafide_pdf(data: dict) -> io.BytesIO:
    from datetime import datetime
    
    buffer = io.BytesIO()
    page_size = landscape(A4)
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size
    
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    if current_month >= 7:
        academic_year = f"{current_year}-{current_year + 1}"
    else:
        academic_year = f"{current_year - 1}-{current_year}"
    
    y = height - 1 * inch
    
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.HexColor('#1a237e'))
    c.drawCentredString(width / 2, y - 0.2 * inch, "NLP INSTITUTE OF TECHNOLOGY")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, y - 0.45 * inch, "Vellore - 999999, Tamil Nadu, India")
    
    y -= 1.1 * inch
    c.setStrokeColor(colors.HexColor('#1a237e'))
    c.setLineWidth(2)
    c.line(1 * inch, y, width - 1 * inch, y)
    
    y -= 0.8 * inch
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor('#1a237e'))
    c.drawCentredString(width / 2, y, "BONAFIDE CERTIFICATE")
    
    y -= 0.5 * inch
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    
    cert_number = f"VIT/ACAD/{data['roll_number']}/{data['date'].split()[-1]}"
    
    c.drawString(1.5 * inch, y, f"Certificate No: {cert_number}")
    
    c.drawRightString(width - 1.5 * inch, y, f"Date: {data['date']}")
    
    y -= 0.8 * inch
    
    c.setFont("Helvetica", 12)
    content_left = 2 * inch
    content_width = width - 4 * inch
    
    text = "This is to certify that"
    c.drawString(content_left, y, text)
    
    y -= 0.7 * inch
    
    details = [
        ["Name", ":", data['name'].upper()],
        ["Registration Number", ":", data['roll_number']],
        ["Program", ":", data['course']],
        ["Current Year of Study", ":", data['year'].title()],
    ]
    
    table_data = []
    for row in details:
        table_data.append(row)
    
    col_widths = [2.2 * inch, 0.3 * inch, 4 * inch]
    table = Table(table_data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 11),
        ('FONT', (1, 0), (1, -1), 'Helvetica', 11),
        ('FONT', (2, 0), (2, -1), 'Helvetica', 11),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    table.wrapOn(c, content_width, height)
    table.drawOn(c, content_left, y - 1.2 * inch)
    
    y -= 2 * inch
    
    c.setFont("Helvetica", 12)
    line1 = f"is a bonafide student of this institution for the academic year {academic_year}."
    c.drawString(content_left, y, line1)
    
    y -= 0.4 * inch
    line2 = f"This certificate is issued for the purpose of {data['purpose']}."
    c.drawString(content_left, y, line2)
    
    y -= 1.2 * inch
    
    signature_y = y
    signature_left = width - 3.5 * inch
    
    c.setLineWidth(1)
    c.line(signature_left, signature_y, width - 1.5 * inch, signature_y)
    
    y -= 0.25 * inch
    c.setFont("Helvetica-Bold", 10)
    c.drawString(signature_left, y, "Authorized Signatory")
    
    y -= 0.2 * inch
    c.setFont("Helvetica", 9)
    c.drawString(signature_left, y, "Assistant Registrar (Academic)")
    
    seal_x = signature_left - 1.5 * inch
    seal_y = signature_y - 0.5 * inch
    c.setStrokeColor(colors.HexColor('#1a237e'))
    c.setLineWidth(2)
    c.circle(seal_x + 0.5 * inch, seal_y, 0.5 * inch, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(seal_x + 0.5 * inch, seal_y - 0.05 * inch, "OFFICIAL")
    c.drawCentredString(seal_x + 0.5 * inch, seal_y - 0.2 * inch, "SEAL")
    
    y = 0.7 * inch
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    
    footer1 = "This is a computer-generated certificate for project demonstration purposes, This certificate does not hold any value in the real world."
    c.drawCentredString(width / 2, y, footer1)
    
    c.setStrokeColor(colors.HexColor('#1a237e'))
    c.setLineWidth(3)
    c.rect(0.7 * inch, 0.5 * inch, width - 1.4 * inch, height - 1 * inch)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    
    return buffer


if __name__ == "__main__":
    from datetime import datetime
    
    test_data = {
        "name": "Prabhath Kumar",
        "roll_number": "22BCE0001",
        "course": "B.Tech Biotechnology",
        "year": "2nd year",
        "purpose": "bank loan application",
        "date": datetime.now().strftime('%B %d, %Y')
    }
    
    pdf_buffer = generate_bonafide_pdf(test_data)
    
    with open("test_bonafide.pdf", "wb") as f:
        f.write(pdf_buffer.read())
    
    print("Test PDF generated: test_bonafide.pdf")