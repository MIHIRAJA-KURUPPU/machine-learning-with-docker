#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from app import create_app

# Create Flask app
app = create_app()

if __name__ == '__main__':
    # Ensure templates directory exists inside the app folder
    os.makedirs(os.path.join('app', 'templates'), exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


