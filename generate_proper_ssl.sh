#!/bin/bash

# Safe SSL Generation Script
# Generates Nginx-compatible SSL files ensuring proper formatting (newlines).

# Directories
SOURCE_DIR="/home/user/ai-vision-platform/certificate/ssl"
DEST_DIR="/home/user/ai-vision-platform/ssl"
BACKUP_DIR="/home/user/ai-vision-platform/ssl_backup_$(date +%Y%m%d_%H%M%S)"

# Files
SERVER_CERT="$SOURCE_DIR/m-devsecops.com.cer"
CA_CERT="$SOURCE_DIR/CloudflareCA.cer"
KEY_SOURCE="$SOURCE_DIR/m-devsecops.com.key"
PFX_FILE="$SOURCE_DIR/m-devsecops.com.pfx"

# Output Files
FULLCHAIN="$DEST_DIR/fullchain.pem"
CA_CHAIN="$DEST_DIR/ca-chain.crt"
KEY_FILE="$DEST_DIR/key.pem"
CERT_FILE="$DEST_DIR/cert.pem"

echo "=== Starting Safe SSL Generation ==="
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"

# 1. Create Destination and Backup
echo "Creating backup of existing SSL folder..."
if [ -d "$DEST_DIR" ]; then
    cp -r "$DEST_DIR" "$BACKUP_DIR"
    echo "Backup created at $BACKUP_DIR"
else
    mkdir -p "$DEST_DIR"
fi

# 2. Function to convert to PEM if needed and print with guaranteed newline
process_cert() {
    local file=$1
    local name=$2

    if [ ! -f "$file" ]; then
        echo "❌ Error: $name file not found at $file" >&2
        return 1
    fi

    echo "Processing $name..." >&2

    # Check if already PEM format
    if openssl x509 -in "$file" -text -noout >/dev/null 2>&1; then
        # Already PEM
        cat "$file"
    else
        # Try converting DER to PEM
        echo "  (Converting DER to PEM)" >&2
        openssl x509 -inform DER -in "$file" -outform PEM
    fi

    # Always append a newline to be safe
    echo ""
}

# 3. Generate cert.pem (Server cert only)
echo "------------------------------------------------"
echo "Generating cert.pem (Server Certificate only)..."
process_cert "$SERVER_CERT" "Server Certificate" > "$CERT_FILE"

if [ -s "$CERT_FILE" ]; then
    echo "✅ cert.pem created."
else
    echo "❌ Failed to create cert.pem"
    exit 1
fi

# 4. Generate fullchain.pem (Server + CA)
echo "------------------------------------------------"
echo "Generating fullchain.pem (Server + CloudFlare CA)..."
{
    process_cert "$SERVER_CERT" "Server Certificate"
    process_cert "$CA_CERT" "CloudFlare CA Certificate"
} > "$FULLCHAIN"

if [ -s "$FULLCHAIN" ]; then
    echo "✅ fullchain.pem created."
else
    echo "❌ Failed to create fullchain.pem"
    exit 1
fi

# 5. Generate ca-chain.crt (CA Certificate)
echo "------------------------------------------------"
echo "Generating ca-chain.crt (CloudFlare CA)..."
process_cert "$CA_CERT" "CloudFlare CA Certificate" > "$CA_CHAIN"

if [ -s "$CA_CHAIN" ]; then
    echo "✅ ca-chain.crt created."
else
    echo "❌ Failed to create ca-chain.crt"
    exit 1
fi

# 6. Handle Private Key
echo "------------------------------------------------"
echo "Handling Private Key..."

# Strategy A: Use the direct key file from source
if [ -f "$KEY_SOURCE" ]; then
    echo "Using key file from source: $KEY_SOURCE"
    # Convert to traditional RSA PEM if it's PKCS#8 (for broader Nginx compatibility)
    if grep -q "BEGIN PRIVATE KEY" "$KEY_SOURCE"; then
        echo "  (Converting PKCS#8 to RSA PEM format for Nginx compatibility)"
        openssl rsa -in "$KEY_SOURCE" -out "$KEY_FILE" 2>/dev/null
        if [ ! -s "$KEY_FILE" ]; then
            # If RSA conversion fails (e.g. EC key), just copy as-is
            echo "  (RSA conversion failed, copying as-is — PKCS#8 also works with modern Nginx)"
            cp "$KEY_SOURCE" "$KEY_FILE"
        fi
    else
        cp "$KEY_SOURCE" "$KEY_FILE"
    fi

    if [ -s "$KEY_FILE" ]; then
        echo "✅ key.pem set from source key file."
    else
        echo "❌ Failed to set key.pem from source."
    fi

# Strategy B: Extract from PFX if direct key not available
elif [ -f "$PFX_FILE" ]; then
    echo "Extracting private key from PFX..."
    openssl pkcs12 -in "$PFX_FILE" -nocerts -nodes -out "$KEY_FILE" -passin pass: 2>/dev/null || \
    openssl pkcs12 -in "$PFX_FILE" -nocerts -nodes -out "$KEY_FILE"

    if [ -s "$KEY_FILE" ]; then
        echo "✅ Key extracted from PFX."
    else
        echo "❌ Failed to extract key from PFX (password might be required)."
    fi
else
    echo "⚠️  No private key found (checked source key file and PFX)."
fi

# 7. Set Permissions
chmod 644 "$FULLCHAIN" "$CA_CHAIN" "$CERT_FILE"
[ -f "$KEY_FILE" ] && chmod 600 "$KEY_FILE"

# 8. Verification
echo "------------------------------------------------"
echo "Verifying Certificates..."

# Check fullchain
if openssl x509 -in "$FULLCHAIN" -noout 2>/dev/null; then
    echo "✅ fullchain.pem format is VALID."
    openssl x509 -in "$FULLCHAIN" -noout -subject -dates
else
    echo "❌ fullchain.pem format is INVALID."
fi

# Check ca-chain
if openssl x509 -in "$CA_CHAIN" -noout 2>/dev/null; then
    echo "✅ ca-chain.crt format is VALID."
else
    echo "❌ ca-chain.crt format is INVALID."
fi

# Check key
if [ -f "$KEY_FILE" ]; then
    if openssl pkey -in "$KEY_FILE" -noout 2>/dev/null; then
        echo "✅ key.pem is VALID."
    else
        echo "❌ key.pem is INVALID."
    fi
fi

# Check cert/key match
if [ -f "$KEY_FILE" ] && [ -f "$FULLCHAIN" ]; then
    CERT_PUB=$(openssl x509 -in "$FULLCHAIN" -noout -pubkey 2>/dev/null | openssl md5)
    KEY_PUB=$(openssl pkey -in "$KEY_FILE" -pubout 2>/dev/null | openssl md5)

    if [ "$CERT_PUB" == "$KEY_PUB" ]; then
        echo "✅ Certificate and Key MATCH."
    else
        echo "❌ Certificate and Key DO NOT MATCH!"
        echo "Cert pubkey MD5: $CERT_PUB"
        echo "Key pubkey MD5:  $KEY_PUB"
    fi
fi

echo ""
echo "=== SSL Files Summary ==="
ls -lh "$DEST_DIR/"
echo "=== Done ==="
