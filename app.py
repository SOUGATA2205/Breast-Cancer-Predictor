from flask import Flask, request, render_template, make_response, session, url_for, redirect, flash, jsonify
import pickle
import numpy as np
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import datetime
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # For session management

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# Create database if it doesn't exist
init_db()

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
print("Model loaded successfully!")

# Authentication Helper Functions
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_strong_password(password):
    # At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    return len(password) >= 8 and any(c.isupper() for c in password) and any(c.islower() for c in password) and any(c.isdigit() for c in password)

def get_user_by_email(email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def login_required(view):
    def wrapped_view(**kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return view(**kwargs)
    wrapped_view.__name__ = view.__name__
    return wrapped_view

# Routes
@app.route('/')
def home():
    return render_template('landing.html', 
                           user_name=session.get('user_name'),
                           user_role=session.get('user_role'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role')
        terms = request.form.get('terms')
        
        # Validation
        if not name or not email or not password or not confirm_password or not role:
            return render_template('signup.html', error='All fields are required')
        
        if not is_valid_email(email):
            return render_template('signup.html', error='Please enter a valid email address')
        
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')
        
        if not is_strong_password(password):
            return render_template('signup.html', error='Password must be at least 8 characters long and include uppercase, lowercase, and numbers')
        
        if not terms:
            return render_template('signup.html', error='You must agree to the Terms of Service')
        
        # Check if user already exists
        existing_user = get_user_by_email(email)
        if existing_user:
            return render_template('signup.html', error='Email is already registered')
        
        # Create the user
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
            (name, email, generate_password_hash(password), role)
        )
        conn.commit()
        conn.close()
        
        return redirect(url_for('login', success='Account created successfully! Please log in.'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember')
        
        # Validation
        if not email or not password:
            return render_template('login.html', error='Please enter both email and password')
        
        # Check user credentials
        user = get_user_by_email(email)
        if not user or not check_password_hash(user[3], password):
            return render_template('login.html', error='Invalid email or password')
        
        # Log in user
        session.clear()
        session['user_id'] = user[0]
        session['user_name'] = user[1]
        session['user_email'] = user[2]
        session['user_role'] = user[4]
        
        # Set session expiration
        if remember:
            session.permanent = True  # Will expire after 31 days by default
        
        next_url = request.args.get('next')
        if next_url:
            return redirect(next_url)
        return redirect(url_for('tool'))
    
    success = request.args.get('success')
    return render_template('login.html', success=success)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/tool')
@login_required
def tool():
    return render_template('index.html', user_name=session.get('user_name'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Get patient details
    patient_details = {
        'name': request.form.get('patient_name'),
        'age': request.form.get('patient_age'),
        'gender': request.form.get('patient_gender'),
        'contact': request.form.get('patient_contact'),
        'blood_group': request.form.get('patient_blood'),
        'email': request.form.get('patient_email')
    }
    
    # Get feature values from the form
    features = [
        float(request.form.get('radius_mean')),
        float(request.form.get('texture_mean')),
        float(request.form.get('perimeter_mean')),
        float(request.form.get('area_mean')),
        float(request.form.get('smoothness_mean')),
        float(request.form.get('compactness_mean')),
        float(request.form.get('concavity_mean')),
        float(request.form.get('concave_points_mean')),
        float(request.form.get('symmetry_mean')),
        float(request.form.get('fractal_dimension_mean')),
        float(request.form.get('radius_se')),
        float(request.form.get('texture_se')),
        float(request.form.get('perimeter_se')),
        float(request.form.get('area_se')),
        float(request.form.get('smoothness_se')),
        float(request.form.get('compactness_se')),
        float(request.form.get('concavity_se')),
        float(request.form.get('concave_points_se')),
        float(request.form.get('symmetry_se')),
        float(request.form.get('fractal_dimension_se')),
        float(request.form.get('radius_worst')),
        float(request.form.get('texture_worst')),
        float(request.form.get('perimeter_worst')),
        float(request.form.get('area_worst')),
        float(request.form.get('smoothness_worst')),
        float(request.form.get('compactness_worst')),
        float(request.form.get('concavity_worst')),
        float(request.form.get('concave_points_worst')),
        float(request.form.get('symmetry_worst')),
        float(request.form.get('fractal_dimension_worst'))
    ]
    
    # Add 0 for the ID feature since it's in the model but not used for prediction
    final_features = [0] + features  # Add a placeholder for the 'id' column
    
    # Make prediction
    prediction = model.predict([final_features])
    probability = model.predict_proba([final_features])
    
    # Determine result
    result = "Malignant (M)" if prediction[0] == 'M' else "Benign (B)"
    confidence = probability[0][1] if prediction[0] == 'M' else probability[0][0]
    
    # Store prediction results and patient details in session for PDF generation
    session['prediction'] = {
        'result': result,
        'confidence': f'{confidence:.2%}',
        'patient': patient_details,
        'features': features,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'doctor': session.get('user_name')  # Add the doctor's name who made the prediction
    }
    
    return render_template('index.html', 
                          prediction_text=f'Breast Cancer Prediction: {result}',
                          confidence=f'Confidence: {confidence:.2%}',
                          patient_name=patient_details['name'],
                          patient_age=patient_details['age'],
                          patient_gender=patient_details['gender'],
                          patient_contact=patient_details['contact'],
                          patient_blood=patient_details['blood_group'],
                          patient_email=patient_details['email'],
                          user_name=session.get('user_name'))

@app.route('/generate_pdf')
@login_required
def generate_pdf():
    # Get prediction results from session
    prediction_data = session.get('prediction', {})
    
    # Get patient details from session instead of query parameters
    patient_details = prediction_data.get('patient', {})
    patient_name = patient_details.get('name', request.args.get('name', 'N/A'))
    patient_age = patient_details.get('age', request.args.get('age', 'N/A'))
    patient_gender = patient_details.get('gender', request.args.get('gender', 'N/A'))
    patient_contact = patient_details.get('contact', request.args.get('contact', 'N/A'))
    patient_blood = patient_details.get('blood_group', request.args.get('blood', 'N/A'))
    patient_email = patient_details.get('email', request.args.get('email', 'N/A'))
    
    # Get prediction results
    result = prediction_data.get('result', 'N/A')
    confidence = prediction_data.get('confidence', 'N/A')
    timestamp = prediction_data.get('timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    doctor_name = prediction_data.get('doctor', session.get('user_name', 'N/A'))
    
    # Create a PDF buffer
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = styles['Title']
    elements.append(Paragraph("Breast Cancer Prediction Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Add timestamp and doctor info
    elements.append(Paragraph(f"Report Generated: {timestamp}", styles['Normal']))
    elements.append(Paragraph(f"Generated By: {doctor_name}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add patient information
    elements.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [
        ["Name:", patient_name],
        ["Age:", patient_age],
        ["Gender:", patient_gender],
        ["Contact:", patient_contact],
        ["Blood Group:", patient_blood],
        ["Email:", patient_email]
    ]
    
    patient_table = Table(patient_data, colWidths=[100, 300])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Add prediction results
    elements.append(Paragraph("Prediction Results", styles['Heading2']))
    
    # Set color based on result
    result_color = colors.red if "Malignant" in result else colors.green
    
    prediction_data = [
        ["Diagnosis:", result],
        ["Confidence:", confidence]
    ]
    
    prediction_table = Table(prediction_data, colWidths=[100, 300])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('TEXTCOLOR', (1, 0), (1, 0), result_color),  # Color the result
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(prediction_table)
    elements.append(Spacer(1, 20))
    
    # Add disclaimer
    elements.append(Paragraph("Disclaimer", styles['Heading3']))
    elements.append(Paragraph(
        "This report is generated based on machine learning predictions and should be reviewed by a healthcare professional. "
        "The prediction is not a definitive diagnosis and should be used only as a supporting tool for medical professionals.",
        styles['Normal']
    ))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    # Create response
    response = make_response(pdf_value)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=breast_cancer_report_{patient_name.replace(" ", "_")}.pdf'
    
    return response

@app.route('/send_email_report', methods=['POST'])
@login_required
def send_email_report():
    try:
        # Get prediction results from session
        prediction_data = session.get('prediction', {})
        
        if not prediction_data:
            return jsonify({"success": False, "message": "No prediction data available. Please make a prediction first."}), 400
        
        # Get patient details
        patient_details = prediction_data.get('patient', {})
        patient_email = patient_details.get('email')
        patient_name = patient_details.get('name')
        
        if not patient_email:
            return jsonify({"success": False, "message": "Patient email not provided. Please include patient email."}), 400
        
        # Generate PDF
        try:
            pdf_data = generate_pdf_data()
        except Exception as e:
            return jsonify({"success": False, "message": f"Failed to generate PDF: {str(e)}"}), 500
        
        # Real email configuration
        sender_email = "automatedbreastcancer001@gmail.com"
        sender_password = "fcmv suxo cuks igfg"  # App password
        sender_name = "Breast Cancer Predictor"
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        # Prepare email
        message = MIMEMultipart()
        message['From'] = f"{sender_name} <{sender_email}>"
        message['To'] = patient_email
        message['Subject'] = f"Breast Cancer Prediction Report - {patient_name}"
        
        # Email body
        doctor_name = prediction_data.get('doctor', 'Your doctor')
        result = prediction_data.get('result', '')
        is_malignant = "Malignant" in result
        
        email_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border-left: 5px solid {'#e74c3c' if is_malignant else '#27ae60'};">
                <h2 style="color: #2c3e50;">Breast Cancer Prediction Report</h2>
                <p>Dear {patient_name},</p>
                <p>Please find attached your breast cancer prediction report as prepared by Dr. {doctor_name}.</p>
                <p>The prediction result indicates: <strong style="color: {'#e74c3c' if is_malignant else '#27ae60'};">{result}</strong></p>
                <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 30px; border-top: 1px solid #eee; padding-top: 15px;">
                    This is an automated message. Please do not reply to this email.<br>
                    For any concerns regarding your report, please contact your healthcare provider.
                </p>
            </div>
        </body>
        </html>
        """
        
        # Attach HTML content
        message.attach(MIMEText(email_body, 'html'))
        
        # Attach PDF
        attachment = MIMEApplication(pdf_data)
        attachment.add_header('Content-Disposition', 'attachment', 
                             filename=f"breast_cancer_report_{patient_name.replace(' ', '_')}.pdf")
        message.attach(attachment)
        
        # Send email
        try:
            # Connect to SMTP server and send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)
            
            # Log successful email
            print(f"Email sent successfully to {patient_email}")
            
            # Return success response
            return jsonify({
                "success": True, 
                "message": f"Report sent to {patient_email} successfully"
            }), 200
            
        except Exception as e:
            print(f"SMTP Error: {str(e)}")
            return jsonify({"success": False, "message": f"Failed to send email: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Email error: {str(e)}")
        return jsonify({"success": False, "message": "An error occurred while sending the email. Please try again."}), 500

def generate_pdf_data():
    """Generate PDF data for email attachment"""
    # Get prediction results from session
    prediction_data = session.get('prediction', {})
    
    # Get patient details from session
    patient_details = prediction_data.get('patient', {})
    patient_name = patient_details.get('name', 'N/A')
    patient_age = patient_details.get('age', 'N/A')
    patient_gender = patient_details.get('gender', 'N/A')
    patient_contact = patient_details.get('contact', 'N/A')
    patient_blood = patient_details.get('blood_group', 'N/A')
    patient_email = patient_details.get('email', 'N/A')
    
    # Get prediction results
    result = prediction_data.get('result', 'N/A')
    confidence = prediction_data.get('confidence', 'N/A')
    timestamp = prediction_data.get('timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    doctor_name = prediction_data.get('doctor', 'N/A')
    
    # Create a PDF buffer
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = styles['Title']
    elements.append(Paragraph("Breast Cancer Prediction Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Add timestamp and doctor info
    elements.append(Paragraph(f"Report Generated: {timestamp}", styles['Normal']))
    elements.append(Paragraph(f"Generated By: {doctor_name}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add patient information
    elements.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [
        ["Name:", patient_name],
        ["Age:", patient_age],
        ["Gender:", patient_gender],
        ["Contact:", patient_contact],
        ["Blood Group:", patient_blood],
        ["Email:", patient_email]
    ]
    
    patient_table = Table(patient_data, colWidths=[100, 300])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Add prediction results
    elements.append(Paragraph("Prediction Results", styles['Heading2']))
    
    # Set color based on result
    result_color = colors.red if "Malignant" in result else colors.green
    
    prediction_data = [
        ["Diagnosis:", result],
        ["Confidence:", confidence]
    ]
    
    prediction_table = Table(prediction_data, colWidths=[100, 300])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('TEXTCOLOR', (1, 0), (1, 0), result_color),  # Color the result
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(prediction_table)
    elements.append(Spacer(1, 20))
    
    # Add disclaimer
    elements.append(Paragraph("Disclaimer", styles['Heading3']))
    elements.append(Paragraph(
        "This report is generated based on machine learning predictions and should be reviewed by a healthcare professional. "
        "The prediction is not a definitive diagnosis and should be used only as a supporting tool for medical professionals.",
        styles['Normal']
    ))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    return pdf_value

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    # Get user information
    user_id = session.get('user_id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    # Handle form submission for profile update
    message = None
    error = None
    
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        if not name or not email:
            error = 'Name and email are required fields'
        elif not is_valid_email(email):
            error = 'Please enter a valid email address'
        elif email != user[2] and get_user_by_email(email):  # Check if new email is already used
            error = 'This email is already registered with another account'
        elif current_password and not check_password_hash(user[3], current_password):
            error = 'Current password is incorrect'
        elif new_password and not confirm_password:
            error = 'Please confirm your new password'
        elif new_password and new_password != confirm_password:
            error = 'New passwords do not match'
        elif new_password and not is_strong_password(new_password):
            error = 'New password must be at least 8 characters long and include uppercase, lowercase, and numbers'
        else:
            # Update user information
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            # Update name and email
            cursor.execute('UPDATE users SET name = ?, email = ? WHERE id = ?', (name, email, user_id))
            
            # Update password if provided
            if new_password:
                hashed_password = generate_password_hash(new_password)
                cursor.execute('UPDATE users SET password = ? WHERE id = ?', (hashed_password, user_id))
            
            conn.commit()
            conn.close()
            
            # Update session data
            session['user_name'] = name
            session['user_email'] = email
            
            message = 'Profile updated successfully'
            
            # Refresh user data
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            conn.close()
    
    return render_template('profile.html', 
                          user_name=user[1], 
                          user_email=user[2], 
                          user_role=user[4], 
                          user_created=user[5],
                          message=message,
                          error=error)

if __name__ == '__main__':
    app.run(debug=True)