import streamlit as st
from langserve import RemoteRunnable

# Use Markdown and styling to enhance the title with icon and gradient colors
st.markdown("""
<h1 style='text-align: center; color: blue; background: linear-gradient(to right, red, purple); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
      🎬 YouTube Real-Time Data Analysis
</h1>
""", unsafe_allow_html=True)


# Input field
input_text = st.text_input("Please enter your question:", key="2")

if input_text:
    with st.spinner("Processing..."):
        try:
            app = RemoteRunnable("http://localhost:8000/youtube_agent_chat")
            responses = []
            for output in app.stream({"input": input_text}):
                responses.append(output)
            if responses:
                st.subheader('Analysis Results')
                last_response = responses[-1]
                st.markdown(last_response["generate"]["generation"])

                # Collapsible display of documents content
                with st.expander("View Detailed Recommended Video Information"):
                    for idx, doc in enumerate(last_response.get("documents", [])):
                        st.write(f"### Video {idx + 1}")
                        st.json(doc)  # Display detailed content of each document
            else:
                st.info("No results returned.")
        except Exception as e:
            st.error(f"Error occurred during processing: {str(e)}")


