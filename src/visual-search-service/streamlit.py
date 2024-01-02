import streamlit as st
import time
from PIL import Image
import requests
from io import BytesIO
from services import ImageSearchService



@st.cache_resource
def get_image_search_service():
    service = ImageSearchService()
    return service

def display_job_statuses(service, state):
    if 'shown_jobs' not in state:
        state.shown_jobs = set()

    for job_id, job_data in service.jobs.items():
        unique_key = f"{job_id}_{job_data['jobStatus']}"
        
        # Remove job_id from shown_jobs if status has changed.
        prev_key = next((k for k in state.shown_jobs if k.startswith(f"{job_id}_")), None)
        if prev_key and prev_key != unique_key:
            state.shown_jobs.discard(prev_key)

        # Display the job only if it's not in the shown_jobs set.
        if unique_key not in state.shown_jobs:
            state.shown_jobs.add(unique_key)
            st.write(f"Job {job_id}: {job_data['jobStatus']}")

            if 'productItems' in job_data:
                st.write("Matched Products:")
                for product in job_data['productItems']:
                    with st.expander(f"{product['product_name']}"):
                        cols = st.columns((1, 2, 2))
                        
                        # Display product image
                        if product['image']:
                            response = requests.get(product['image'][0])
                            img = Image.open(BytesIO(response.content))
                            cols[0].image(img, use_column_width=True)

                        # Display product details
                        cols[1].markdown(f"**Price:** {product['retail_price']} (Discounted Price: {product['discounted_price']})")
                        cols[1].markdown(f"**Rating:** {product['product_rating']}")
                        cols[1].markdown(f"**Description:** {product['generated_description']}")
                        
                        # Display product link
                        cols[2].markdown(f"[View on Flipkart]({product['product_url']})")


def main():
    st.title('Visual Search Demo')

    if 'service' not in st.session_state:
        st.session_state['service'] = get_image_search_service()

    service = st.session_state['service']

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    jobs_placeholder = st.empty()  # Create a placeholder for job statuses

    if uploaded_file is not None:
        # Check if the file has already been submitted
        if 'last_uploaded_file' in st.session_state and st.session_state['last_uploaded_file'] == uploaded_file:
            pass  # Do nothing if this file has already been submitted
        else:
            # Save this file to session state and submit the job
            st.session_state['last_uploaded_file'] = uploaded_file
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            service.search_image(uploaded_file)
            
        if 'shown_jobs' not in st.session_state:
            st.session_state.shown_jobs = set()

        while True:
            display_job_statuses(service, st.session_state)

            time.sleep(2)

if __name__ == "__main__":
    main()
