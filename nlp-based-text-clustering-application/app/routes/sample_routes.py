from flask import Blueprint
import pandas as pd
from io import BytesIO
from flask import send_file

# Create a Blueprint
sample_bp = Blueprint('sample', __name__, url_prefix='/sample')

@sample_bp.route('', methods=['GET'])
def get_sample():
    """Provide a sample dataset for testing"""
    # Create sample data
    data = {
        'text': [
            "Machine learning is a subfield of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "Deep learning models consist of multiple layers of neural networks that can recognize patterns in vast amounts of data.",
            "Neural networks are inspired by the human brain and consist of interconnected nodes that process information.",
            "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment and receiving rewards.",
            "Natural language processing combines computational linguistics and machine learning to process and analyze text data.",
            "Python is one of the most popular programming languages used in data science and machine learning.",
            "JavaScript is primarily used for web development but is expanding into machine learning with libraries like TensorFlow.js.",
            "R programming language is widely used for statistical analysis and data visualization in research.",
            "SQL is essential for data scientists to extract and manipulate data from relational databases.",
            "Rust provides memory safety without garbage collection, making it efficient for system programming.",
            "Data mining techniques help discover patterns and extract useful information from large datasets.",
            "Feature engineering is the process of selecting and transforming variables when creating a predictive model.",
            "Regression analysis helps understand relationships between dependent and independent variables in datasets.",
            "Cluster analysis groups objects based on their similarity, useful for market segmentation.",
            "Principal component analysis reduces the dimensionality of data while preserving important information.",
            "Climate change is accelerating with rising global temperatures affecting ecosystems worldwide.",
            "Renewable energy sources like solar and wind are becoming increasingly cost-competitive with fossil fuels.",
            "Biodiversity loss threatens ecosystem stability and human well-being across the planet.",
            "Water scarcity affects billions of people and is worsening due to climate change and population growth.",
            "Sustainable agriculture practices can reduce environmental impact while maintaining crop yields.",
            "The healthcare industry is implementing AI for early disease detection and personalized treatment plans.",
            "Telemedicine has expanded rapidly, allowing patients to consult with healthcare providers remotely.",
            "Electronic health records improve patient care coordination but raise privacy concerns.",
            "Preventive healthcare focuses on disease prevention rather than treatment, reducing overall costs.",
            "Genomic medicine uses information about a person's genetic makeup to tailor medical treatments.",
            "E-commerce has transformed retail by allowing businesses to reach customers globally online.",
            "Supply chain management optimizes the flow of goods and services from raw materials to finished products.",
            "Digital marketing uses online channels to connect with customers and promote products or services.",
            "Remote work has increased productivity for many companies while reducing overhead costs.",
            "Blockchain technology offers transparent and secure transactions for businesses across industries.",
            "Virtual reality creates immersive experiences for gaming, education, and professional training.",
            "Quantum computing promises to solve complex problems that are beyond the capabilities of classical computers.",
            "Internet of Things connects everyday devices to the internet, enabling data collection and remote control.",
            "Cybersecurity measures protect systems and networks from digital attacks and unauthorized access.",
            "Cloud computing provides on-demand access to computing resources without direct active management.",
            "Social media platforms have changed how people communicate and share information globally.",
            "Digital literacy is increasingly important as technology becomes integrated into daily life.",
            "Visual communication using graphics and imagery can convey complex ideas more effectively than text alone.",
            "Intercultural communication skills are essential in the globalized business environment.",
            "Nonverbal communication includes facial expressions, gestures, and body language that convey meaning."
        ],
        'category': [
            "AI", "AI", "AI", "AI", "AI",
            "Programming", "Programming", "Programming", "Programming", "Programming",
            "Data Science", "Data Science", "Data Science", "Data Science", "Data Science",
            "Environment", "Environment", "Environment", "Environment", "Environment",
            "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare",
            "Business", "Business", "Business", "Business", "Business",
            "Technology", "Technology", "Technology", "Technology", "Technology",
            "Communication", "Communication", "Communication", "Communication", "Communication"
        ],
        'source': [
            "Tech Blog", "Research Paper", "Textbook", "Course Materials", "Conference",
            "Developer Forum", "Tech Article", "Academic Source", "Tutorial", "Documentation",
            "Research Journal", "Online Course", "Statistics Textbook", "Business Report", "Analytics Blog",
            "Scientific Journal", "News Article", "Conservation Report", "UN Publication", "Agricultural Study",
            "Medical Journal", "Health Magazine", "Policy Brief", "Public Health Report", "Research Paper",
            "Business Review", "Logistics Journal", "Marketing Guide", "Workplace Study", "Financial Report",
            "Tech Magazine", "Scientific American", "Engineering Journal", "Security Report", "IT Publication",
            "Social Science Journal", "Education Report", "Design Publication", "International Relations", "Psychology Text"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create CSV file
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        download_name='sample_text_data.csv',
        as_attachment=True
    )
