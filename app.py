from flask import Flask, request, jsonify, render_template
import os
import base64
import json
import logging
import traceback
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("No OpenAI API key found in environment variables")
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)
logger.info("OpenAI client initialized")

@app.route('/')
def index():
    logger.debug("Serving index page")
    return render_template("index.html")

@app.route('/extract', methods=['POST'])
def extract_text():
    request_start_time = time.time()
    logger.debug("Received /extract request")
    
    try:
        # Get the image data from the request
        data = request.json
        if not data:
            logger.error("No JSON data in request")
            return jsonify({"status": "error", "message": "No JSON data provided"})
        
        logger.debug(f"Request data keys: {list(data.keys())}")
        
        if 'image' not in data:
            logger.error("No image key in request data")
            return jsonify({"status": "error", "message": "No image data provided"})
        
        # The image comes as a base64 string
        image_data = data['image']
        logger.debug(f"Received image data of length: {len(image_data)}")
        
        # If the image starts with the data URL prefix, remove it
        if image_data.startswith('data:image'):
            logger.debug("Image data contains prefix, removing it")
            image_data = image_data.split(',')[1]
        
        # Force extraction flag
        force_extraction = data.get('force_extraction', False)
        
        # Step 1: Use GPT-4o to detect license, crop conceptually, and extract text
        logger.info("Sending image to OpenAI for license detection and text extraction")
        extraction_start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a specialized assistant that extracts text from driver's license images.
                        
                        First, identify the driver's license in the image - it will be a rectangular card with text and possibly a photo.
                        Even if the license only takes up a small portion of the image or has a busy background, focus only on the license.
                        
                        Once you've located the license in the image:
                        1. Extract all visible text from ONLY the license portion
                        2. Ignore any text that is not on the license itself
                        3. Format the extracted text clearly
                        
                        If you cannot find a driver's license in the image, respond with only: "NO_LICENSE_DETECTED"
                        """
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from the driver's license in this image, ignoring any background:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=500
            )
            extraction_time = time.time() - extraction_start_time
            logger.debug(f"Text extraction completed in {extraction_time:.2f} seconds")
            
            # Extract the response text
            extracted_text = response.choices[0].message.content
            logger.info(f"Extracted text ({len(extracted_text)} characters)")
            logger.debug(f"Extracted text content: {extracted_text}")
            
            # Check if no license was detected
            if extracted_text.strip() == "NO_LICENSE_DETECTED" and not force_extraction:
                # Return feedback to help user take a better photo
                suggestions = [
                    "Make sure your driver's license is visible in the image",
                    "Ensure good lighting with minimal glare",
                    "Hold the license parallel to the camera",
                    "Use a contrasting background"
                ]
                
                return jsonify({
                    "status": "error",
                    "message": "Could not clearly detect a driver's license in the image",
                    "analysis": {
                        "license_detected": False
                    },
                    "suggestions": suggestions
                })
            
        except Exception as e:
            logger.error(f"Error in OpenAI text extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"status": "error", "message": f"OpenAI text extraction failed: {str(e)}"})
        
        # Step 2: Process the extracted text to identify driver's license information
        logger.info("Processing extracted text for license information")
        license_start_time = time.time()
        
        try:
            license_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a specialized assistant that extracts driver's license information from text.
                        Analyze the following text and extract structured information for these fields if present:
                        - LIC# (License Number)
                        - Name (Full name as it appears)
                        - DOB (Date of Birth)
                        - Issue Date
                        - Expiration Date
                        - Address (Full address including city, state, zip)
                        - Sex
                        - Height
                        - Weight
                        - Eyes (Eye color)
                        - Restriction
                        - Class (License class)
                        - DD# (Document Discriminator Number)
                        - Donor status
                        - Revision date
                        
                        Format your response as key-value pairs with a colon between the key and value, one per line.
                        If you can't find information for a field, don't include it.
                        Don't make up information or guess. Extract only what's clearly present in the text."""
                    },
                    {
                        "role": "user",
                        "content": f"Extract driver's license information from this text:\n\n{extracted_text}"
                    }
                ],
                max_tokens=500
            )
            license_time = time.time() - license_start_time
            logger.debug(f"License info extraction completed in {license_time:.2f} seconds")
            
            # Extract the processed license information
            license_info = license_response.choices[0].message.content
            logger.info(f"License info extracted ({len(license_info)} characters)")
            logger.debug(f"License info content: {license_info}")
            
        except Exception as e:
            logger.error(f"Error in OpenAI license info extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error", 
                "message": f"OpenAI license info extraction failed: {str(e)}",
                "raw_text": extracted_text  # Return the raw text at least
            })
        
        # Process the license info text into a structured format
        logger.debug("Parsing license info into structured format")
        lines = license_info.split('\n')
        formatted_data = {}
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                formatted_data[key] = value
                logger.debug(f"Parsed field: {key} = {value}")
        
        # If no structured data was found, use the extracted raw text
        if not formatted_data:
            logger.warning("No structured license data found, using raw text")
            formatted_data = {"Raw Extracted Text": extracted_text}
        
        # Log the final result
        total_time = time.time() - request_start_time
        logger.info(f"Total processing completed in {total_time:.2f} seconds")
        logger.info(f"Returning {len(formatted_data)} fields of license data")
        
        return jsonify({
            "status": "success", 
            "data": formatted_data,
            "raw_text": extracted_text,
            "license_info": license_info,
            "processing_time": {
                "extraction_time": f"{extraction_time:.2f}s",
                "license_processing_time": f"{license_time:.2f}s",
                "total_time": f"{total_time:.2f}s"
            }
        })
        
    except Exception as e:
        logger.error(f"Unhandled exception in /extract endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"})

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)
