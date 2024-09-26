# People Detection Project

## Project Overview
The People Detection project is designed to identify and track individuals in real-time using the YOLOv8 model. The system provides accurate and efficient detection, which can be applied in various scenarios such as security monitoring, crowd analysis, and resource management. This project is developed and deployed on Hugging Face Spaces, making it easily accessible for demonstrations and further development.

## Deployment
The project is deployed and can be accessed at:  
[People Detection on Hugging Face](https://huggingface.co/spaces/ducdatit2002/people-detection)

## How to Run Locally

1. **Clone the Repository**  
   Clone the project repository to your local machine:
   ```bash
   git clone https://github.com/ducdatit2002/people-detection.git
   cd people-detection
   ```

2. **Install Dependencies**  
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   Start the Streamlit app to begin detecting people in real-time:
   ```bash
   streamlit run app.py
   ```

## Model and Methodology
This project leverages the YOLOv8 model, an advanced object detection algorithm, to detect people in video streams. YOLOv8 offers high precision and speed, making it ideal for real-time applications. The model can be fine-tuned or customized for specific environments and detection needs.

## Contributors
- **Đỗ Lê Quỳnh Mai** - Author
- **Phạm Đức Đạt** - Deployment to Hugging Face

## Future Enhancements
- **Multiple Object Detection**: Extend the model to detect additional objects like vehicles and animals.
- **Integration with IoT Devices**: Connect with IoT devices for automated responses based on detection results.
- **Advanced Analytics**: Implement features for counting people, tracking movements, and generating heatmaps for crowd analysis.

## License
This project is licensed under the MIT License.

## Contact Information
For any inquiries or contributions, please contact the project contributors via the GitHub repository's issue tracker or their respective emails.

We hope this project will contribute to advancements in real-time people detection and its applications in various fields!
