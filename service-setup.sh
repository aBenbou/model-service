#!/bin/bash
set -e

echo "Model Manager Service Setup"
echo "---------------------------"
echo "This script will set up the Model Manager service by configuring AWS credentials and initializing the service."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Configure AWS credentials if not already done
if ! grep -q "AWS_ACCESS_KEY_ID" .env || ! grep -q "AWS_SECRET_ACCESS_KEY" .env; then
    echo "Setting up AWS credentials..."
    
    # Prompt for AWS credentials
    read -p "Enter your AWS Access Key ID: " aws_access_key_id
    read -sp "Enter your AWS Secret Access Key: " aws_secret_access_key
    echo ""
    read -p "Enter your AWS Region (default: us-east-1): " aws_region
    
    # Set default region if not provided
    if [ -z "$aws_region" ]; then
        aws_region="us-east-1"
    fi
    
    # Write credentials to .env file
    echo "AWS_ACCESS_KEY_ID=$aws_access_key_id" >> .env
    echo "AWS_SECRET_ACCESS_KEY=$aws_secret_access_key" >> .env
    echo "AWS_REGION_NAME=$aws_region" >> .env
else
    echo "AWS credentials already configured in .env file."
fi

# Create required directories if they don't exist
mkdir -p configs models scripts

echo ""
echo "Model Manager service setup complete!"
echo ""
echo "You can now start the service with:"
echo "docker-compose up -d"
echo ""
echo "The API will be available at: http://localhost:8000"
echo "Swagger documentation: http://localhost:8000/docs"