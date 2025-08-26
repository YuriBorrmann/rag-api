# app_streamlit.py
import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import io
import sys

# Page configuration
st.set_page_config(
    page_title="RAG System Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URLs
API_BASE_URL = "http://localhost:8000/api"
DOCUMENTS_URL = f"{API_BASE_URL}/documents"
QUESTION_URL = f"{API_BASE_URL}/question"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #000;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #000;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .test-output {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

class RAGInterface:
    """Main class to manage the RAG System interface"""
    
    def __init__(self):
        """Initialize the interface"""
        self.session_state = st.session_state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize the session state"""
        if 'chat_history' not in self.session_state:
            self.session_state.chat_history = []
        if 'documents' not in self.session_state:
            self.session_state.documents = []
        if 'test_results' not in self.session_state:
            self.session_state.test_results = None
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">🧠 RAG System Interface</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with navigation"""
        st.sidebar.title("📋 Menu")
        
        # Page selector
        page = st.sidebar.selectbox(
            "Select an option:",
            ["📤 Document Upload", "💬 Chat", "🧪 Validation Tests", "📊 Dashboard"]
        )
        
        # System information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ System Information")
        
        # API status
        api_status = self._check_api_status()
        if api_status:
            st.sidebar.success("✅ API Online")
        else:
            st.sidebar.error("❌ API Offline")
        
        # Loaded documents
        if self.session_state.documents:
            st.sidebar.markdown(f"**📄 Documents:** {len(self.session_state.documents)}")
        else:
            st.sidebar.markdown("**📄 Documents:** 0")
        
        return page
    
    def _check_api_status(self) -> bool:
        """Check if the API is online"""
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def render_upload_page(self):
        """Render the document upload page"""
        st.header("📤 Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Select PDF files for upload:",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                self._upload_documents(uploaded_files)
    
    def _upload_documents(self, files):
        """Upload documents to the API"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful_uploads = []
        failed_uploads = []
        
        for i, file in enumerate(files):
            try:
                # Prepare file for upload
                files_dict = {'files': (file.name, file, 'application/pdf')}
                
                # Upload
                response = requests.post(DOCUMENTS_URL, files=files_dict)
                
                if response.status_code == 200:
                    result = response.json()
                    successful_uploads.append({
                        'name': file.name,
                        'size': file.size,
                        'chunks': result.get('total_chunks', 0)
                    })
                else:
                    failed_uploads.append(file.name)
                
            except Exception as e:
                failed_uploads.append(file.name)
                st.error(f"Error processing {file.name}: {str(e)}")
            
            # Update progress
            progress = (i + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text = f"Processing... {int(progress * 100)}%"
        
        # Update session state
        if successful_uploads:
            self.session_state.documents.extend(successful_uploads)
        
        # Show results
        st.subheader("📋 Upload Results")
        
        if successful_uploads:
            st.success(f"✅ {len(successful_uploads)} documents processed successfully!")
            
            # Processed documents table
            df_success = pd.DataFrame(successful_uploads)
            st.dataframe(
                df_success,
                column_config={
                    'name': 'File Name',
                    'size': 'Size (bytes)',
                    'chunks': 'Chunks Generated'
                },
                hide_index=True
            )
        
        if failed_uploads:
            st.error(f"❌ {len(failed_uploads)} documents failed:")
            for name in failed_uploads:
                st.text(f"• {name}")
    
    def render_chat_page(self):
        """Render the chat page"""
        st.header("💬 Chat with Documents")
        
        # Check if there are documents
        if not self.session_state.documents:
            st.warning("⚠️ No documents loaded. Please upload documents first!")
            return
        
        # Show available documents
        st.subheader("📄 Available Documents")
        doc_names = [doc['name'] for doc in self.session_state.documents]
        st.write(", ".join(doc_names))
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="Example: What should be done if damage is found when receiving the motor?",
            key="question_input"
        )
        
        # Send button
        if st.button("🚀 Send Question", type="primary") and question:
            self._ask_question(question)
        
        # Chat history
        st.subheader("💬 Conversation History")
        
        for chat in self.session_state.chat_history:
            with st.container():
                if chat['type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {chat['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Assistant:</strong><br>
                        {chat['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show references
                    if chat['references']:
                        with st.expander("📚 Sources"):
                            for i, ref in enumerate(chat['references'], 1):
                                st.markdown(f"**Source {i}:** {ref[:200]}...")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            self.session_state.chat_history = []
            st.success("History cleared!")
    
    def _ask_question(self, question: str):
        """Ask a question to the API"""
        try:
            with st.spinner("Processing question..."):
                payload = {"question": question}
                headers = {"Content-Type": "application/json"}
                response = requests.post(QUESTION_URL, json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add to history
                    self.session_state.chat_history.append({
                        'type': 'user',
                        'content': question
                    })
                    
                    self.session_state.chat_history.append({
                        'type': 'bot',
                        'content': result['answer'],
                        'references': result.get('references', [])
                    })
                    
                    st.success("✅ Question answered!")
                    
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.text(response.text)
                    
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
    
    def render_test_page(self):
        """Render the test page"""
        st.header("🧪 Validation Tests")
        
        # Run tests button
        if st.button("🚀 Run Validation Tests", type="primary"):
            self._run_validation_tests()
        
        # Show results if they exist
        if self.session_state.test_results:
            self._display_test_results()
    
    def _run_validation_tests(self):
        """Run validation tests"""
        with st.spinner("Running validation tests..."):
            try:
                # Capture stdout to get test results
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                try:
                    # Import and run the test script
                    from test_rag import run_tests
                    success = run_tests()
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
                
                # Get the captured output
                output = captured_output.getvalue()
                
                # Store results
                self.session_state.test_results = {
                    'success': success,
                    'output': output,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                if success:
                    st.success("✅ Tests completed successfully!")
                else:
                    st.warning("⚠️ Tests completed with some failures.")
                    
            except Exception as e:
                st.error(f"❌ Error running tests: {str(e)}")
    
    def _display_test_results(self):
        """Display test results"""
        if not self.session_state.test_results:
            return
        
        st.subheader("📊 Test Results")
        
        # Overall status
        result = self.session_state.test_results
        status_icon = "✅" if result['success'] else "⚠️"
        status_text = "PASSED" if result['success'] else "FAILED"
        status_color = "green" if result['success'] else "orange"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-bottom: 2rem;">
            <h3 style="color: {status_color};">{status_icon} Status: {status_text}</h3>
            <p>Tests run at: {result['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed results
        st.subheader("📋 Detailed Test Output")
        st.markdown(f"""
        <div class="test-output">
        {result['output']}
        </div>
        """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render the dashboard with metrics"""
        st.header("📊 Dashboard")
        
        # Check if there are documents
        if not self.session_state.documents:
            st.warning("⚠️ No documents loaded. Please upload documents first!")
            return
        
        # General metrics
        st.subheader("📈 General Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Documents</div>
            </div>
            """.format(len(self.session_state.documents)), unsafe_allow_html=True)
        
        with col2:
            total_chunks = sum(doc['chunks'] for doc in self.session_state.documents)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Chunks</div>
            </div>
            """.format(total_chunks), unsafe_allow_html=True)
        
        with col3:
            total_size = sum(doc['size'] for doc in self.session_state.documents)
            size_mb = total_size / (1024 * 1024)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}</div>
                <div class="metric-label">Size (MB)</div>
            </div>
            """.format(size_mb), unsafe_allow_html=True)
        
        # Document graph
        st.subheader("📊 Document Distribution")
        
        if self.session_state.documents:
            df = pd.DataFrame(self.session_state.documents)
            
            # Chunks per document graph
            fig = px.bar(
                df,
                x='name',
                y='chunks',
                title='Chunks per Document',
                labels={'name': 'Document', 'chunks': 'Chunks'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Document size graph
            fig2 = px.pie(
                df,
                values='size',
                names='name',
                title='Size Distribution'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Chat statistics
        if self.session_state.chat_history:
            st.subheader("💬 Chat Statistics")
            
            user_messages = [msg for msg in self.session_state.chat_history if msg['type'] == 'user']
            bot_messages = [msg for msg in self.session_state.chat_history if msg['type'] == 'bot']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Questions Asked", len(user_messages))
            
            with col2:
                st.metric("Responses Generated", len(bot_messages))
    
    def run(self):
        """Run the main interface"""
        # Render header
        self.render_header()
        
        # Render sidebar and get selected page
        page = self.render_sidebar()
        
        # Render selected page
        if page == "📤 Document Upload":
            self.render_upload_page()
        elif page == "💬 Chat":
            self.render_chat_page()
        elif page == "🧪 Validation Tests":
            self.render_test_page()
        elif page == "📊 Dashboard":
            self.render_dashboard()

# Run the application
if __name__ == "__main__":
    app = RAGInterface()
    app.run()