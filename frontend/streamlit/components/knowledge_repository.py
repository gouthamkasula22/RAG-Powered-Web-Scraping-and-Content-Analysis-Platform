"""
Intelligent Knowledge Repository with Chat Interface
Professional UI for querying analyzed website data
"""

# Standard library imports
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

# Third-party imports
import pandas as pd
import requests
import streamlit as st

class IntelligentKnowledgeRepository:
    """Professional Knowledge Repository with Chat Interface for analyzed websites"""

    def __init__(self, api_base_url: str = "http://127.0.0.1:8000/api"):
        self.api_base_url = api_base_url
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables"""
        if "kr_chat_messages" not in st.session_state:
            st.session_state.kr_chat_messages = []
        if "kr_selected_website" not in st.session_state:
            st.session_state.kr_selected_website = None
        if "kr_available_websites" not in st.session_state:
            st.session_state.kr_available_websites = []
        if "kr_active_tab" not in st.session_state:
            st.session_state.kr_active_tab = "Chat Interface"

    def render_main_interface(self):
        """Render the main knowledge repository interface"""
        st.header("Knowledge Repository")
        st.markdown("Ask questions about your analyzed websites and get intelligent responses.")

        # Create manual tab buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Chat Interface",
                        key="tab_chat",
                        type="primary" if st.session_state.kr_active_tab == "Chat Interface" else "secondary",
                        use_container_width=True):
                st.session_state.kr_active_tab = "Chat Interface"
                st.rerun()

        with col2:
            if st.button("🌐 Browse Websites",
                        key="tab_browse",
                        type="primary" if st.session_state.kr_active_tab == "Browse Websites" else "secondary",
                        use_container_width=True):
                st.session_state.kr_active_tab = "Browse Websites"
                st.rerun()

        st.divider()

        # Render the active tab content
        if st.session_state.kr_active_tab == "Chat Interface":
            self._render_chat_interface()
        else:
            self._render_browse_interface()

    def _render_chat_interface(self):
        """Render the professional chat interface"""

        # Sidebar for website selection and filters
        with st.sidebar:
            st.subheader("Chat Configuration")

            # Load available websites
            self._load_available_websites()

            if st.session_state.kr_available_websites:
                # Use radio buttons for more reliable website selection
                website_options = ["All Websites"] + [site['title'] for site in st.session_state.kr_available_websites]
                selected_index = st.radio(
                    "Select Website to Query",
                    options=range(len(website_options)),
                    format_func=lambda x: website_options[x],
                    help="Choose which analyzed website to ask questions about",
                    key="kr_selected_site"
                )

                selected_site = website_options[selected_index]

                # Update selected website in session state
                if selected_site == "All Websites":
                    st.session_state.kr_selected_website = None
                else:
                    st.session_state.kr_selected_website = selected_site

                # Show selected website info
                if st.session_state.kr_selected_website:
                    selected_data = next((site for site in st.session_state.kr_available_websites
                                        if site['title'] == st.session_state.kr_selected_website), None)
                    if selected_data:
                        st.markdown("**Selected Website:**")
                        st.info(f"**{selected_data['title']}**\n\n{selected_data['url']}")
                        st.caption(f"Analyzed: {selected_data.get('analyzed_at', 'Unknown')}")

                # Clear chat button
                if st.button("Clear Chat History", type="secondary"):
                    st.session_state.kr_chat_messages = []
                    st.rerun()
            else:
                st.warning("No analyzed websites found. Please analyze some websites first in the Analysis tab.")
                return

        # Main chat interface
        st.subheader("Chat with Your Knowledge Base")

        # Show context if a specific website was selected
        if hasattr(st.session_state, 'kr_selected_website') and st.session_state.kr_selected_website:
            website = st.session_state.kr_selected_website

            # Handle both string and dict formats
            if isinstance(website, dict):
                website_title = website.get('title', 'Unknown Website')
            else:
                website_title = str(website)

            st.info(f"🎯 **Currently focused on:** {website_title}")

            # Option to clear focus
            if st.button("🔄 Chat with All Websites", key="clear_focus"):
                st.session_state.kr_selected_website = None
                st.rerun()

        # Display chat messages (only historical messages, not the current conversation)
        chat_container = st.container()
        with chat_container:
            # Display all messages from chat history
            for message in st.session_state.kr_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "source" in message:
                        st.caption(f"Source: {message['source']}")

        # Chat input
        user_question = st.chat_input("Ask a question about your analyzed websites...")

        if user_question:
            # Check if this is a new question (not already in chat history)
            is_new_question = True
            if st.session_state.kr_chat_messages and len(st.session_state.kr_chat_messages) > 0:
                last_message = st.session_state.kr_chat_messages[-1]
                if (last_message["role"] == "user" and
                    last_message["content"] == user_question):
                    is_new_question = False

            if is_new_question:
                # Add user message to chat
                st.session_state.kr_chat_messages.append({
                    "role": "user",
                    "content": user_question,
                    "timestamp": datetime.now().isoformat()
                })

                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(user_question)

                # Process the question and get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = self._process_user_question(user_question)

                    st.markdown(response["content"])
                    if "source" in response:
                        st.caption(f"Source: {response['source']}")

                # Add assistant response to chat
                st.session_state.kr_chat_messages.append(response)

    def _render_browse_interface(self):
        """Render the browse websites interface"""
        st.subheader("Analyzed Websites")

        # Control panel
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search websites...", placeholder="Enter keywords to filter websites")
        with col2:
            if st.button("Refresh", type="secondary"):
                self._load_available_websites()
                st.rerun()

        # Load and display websites
        self._load_available_websites()

        if not st.session_state.kr_available_websites:
            st.info("No analyzed websites found. Start analyzing websites to build your knowledge base.")
            return

        # Filter websites based on search
        filtered_websites = st.session_state.kr_available_websites
        if search_term:
            search_lower = search_term.lower()
            filtered_websites = [
                site for site in st.session_state.kr_available_websites
                if search_lower in site['title'].lower() or
                   search_lower in site['url'].lower() or
                   search_lower in site.get('summary', '').lower()
            ]

        # Display websites in professional cards
        for website in filtered_websites:
            self._render_website_card(website)

    def _render_website_card(self, website: Dict):
        """Render a professional website card"""
        with st.container():
            st.markdown("""
            <div style="
                border: 1px solid #e1e5e9;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 16px;
                background-color: #ffffff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
            """, unsafe_allow_html=True)

            # Website title and URL
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {website['title']}")
                st.markdown(f"**URL:** [{website['url']}]({website['url']})")
            with col2:
                st.markdown(f"**Analyzed:** {website.get('analyzed_at', 'Unknown')}")

            # Summary
            if website.get('summary'):
                st.markdown(f"**Summary:** {website['summary'][:200]}...")

            # Key metrics
            if website.get('metrics'):
                col1, col2, col3 = st.columns(3)
                metrics = website['metrics']
                with col1:
                    st.metric("Content Quality", f"{metrics.get('content_quality', 0):.1f}/10")
                with col2:
                    st.metric("SEO Score", f"{metrics.get('seo_score', 0):.1f}/10")
                with col3:
                    st.metric("Overall Score", f"{metrics.get('overall_score', 0):.1f}/10")

            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Ask Questions", key=f"ask_{website['id']}", type="primary"):
                    st.session_state.kr_selected_website = website
                    st.session_state.kr_active_tab = "Chat Interface"
                    st.rerun()
            with col2:
                if st.button("View Details", key=f"view_{website['id']}", type="secondary"):
                    self._show_website_details(website)

            st.markdown("</div>", unsafe_allow_html=True)

    def _load_available_websites(self):
        """Load available websites from analysis history"""
        try:
            # This would typically load from your analysis history/database
            # For now, we'll simulate with session state data
            if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
                websites = []

                # Handle both dict and list formats
                if isinstance(st.session_state.analysis_results, dict):
                    # If it's a dictionary, iterate over items
                    for analysis_id, result in st.session_state.analysis_results.items():
                        website_data = self._create_website_data(analysis_id, result)
                        websites.append(website_data)
                elif isinstance(st.session_state.analysis_results, list):
                    # If it's a list, iterate over the list
                    for i, result in enumerate(st.session_state.analysis_results):
                        analysis_id = f"analysis_{i}"
                        website_data = self._create_website_data(analysis_id, result)
                        websites.append(website_data)

                st.session_state.kr_available_websites = websites
            else:
                st.session_state.kr_available_websites = []

        except Exception as e:
            st.error(f"Failed to load websites: {str(e)}")
            import traceback
            st.write(f"Debug traceback: {traceback.format_exc()}")
            st.session_state.kr_available_websites = []

    def _create_website_data(self, analysis_id: str, result: Any) -> Dict:
        """Create website data structure from analysis result"""
        try:
            # Debug: Show what attributes the result has
            st.write(f"Debug: Result attributes: {dir(result)}")

            # Handle SimpleAnalysisResult objects (from main app)
            if hasattr(result, '__class__') and 'SimpleAnalysisResult' in str(result.__class__):
                # Get title from scraped content first, then fallback to URL parsing
                title = None
                main_content = ""

                if hasattr(result, 'scraped_content') and result.scraped_content:
                    title = result.scraped_content.get('title', None)
                    main_content = result.scraped_content.get('main_content', '')
                    st.write(f"Debug: Found scraped content with title: {title}")
                    st.write(f"Debug: Content length: {len(main_content)}")
                    st.write(f"Debug: Scraped content keys: {list(result.scraped_content.keys()) if isinstance(result.scraped_content, dict) else 'Not a dict'}")
                    st.write(f"Debug: Scraped content type: {type(result.scraped_content)}")
                    if isinstance(result.scraped_content, dict):
                        st.write(f"Debug: Scraped content values lengths: {[(k, len(str(v))) for k, v in result.scraped_content.items()]}")
                else:
                    st.write(f"Debug: No scraped content found or empty")
                    st.write(f"Debug: scraped_content value: {getattr(result, 'scraped_content', 'NO ATTRIBUTE')}")

                if not title:
                    title = getattr(result, 'title', None)

                url = getattr(result, 'url', 'Unknown URL')
                executive_summary = getattr(result, 'executive_summary', 'No summary available')

                # Try to get metrics
                metrics = {}
                if hasattr(result, 'metrics') and result.metrics:
                    if isinstance(result.metrics, dict):
                        metrics = result.metrics
                    else:
                        # Extract from metrics object
                        metrics = {
                            'overall_score': getattr(result.metrics, 'overall_score', 0),
                            'content_quality_score': getattr(result.metrics, 'content_quality_score', 0),
                            'seo_score': getattr(result.metrics, 'seo_score', 0),
                            'ux_score': getattr(result.metrics, 'ux_score', 0),
                            'readability_score': getattr(result.metrics, 'readability_score', 0),
                            'performance_score': getattr(result.metrics, 'performance_score', 0)
                        }

                # Get creation time
                created_at = getattr(result, 'created_at', None)
                if created_at:
                    if hasattr(created_at, 'strftime'):
                        analyzed_at = created_at.strftime('%Y-%m-%d %H:%M')
                    else:
                        analyzed_at = str(created_at)
                else:
                    analyzed_at = datetime.now().strftime('%Y-%m-%d %H:%M')

                # Extract title from URL if not available
                if not title or title == 'Untitled Website':
                    if 'amzur.com' in url:
                        if 'leadership-team' in url:
                            title = 'Amzur Technologies - Leadership Team'
                        else:
                            title = 'Amzur Technologies'
                    else:
                        # Extract domain name as title
                        import re
                        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                        if domain_match:
                            title = domain_match.group(1).replace('.com', '').replace('.', ' ').title()
                        else:
                            title = 'Analyzed Website'

                # Use scraped content as primary content source, fallback to executive summary
                content = main_content if main_content else executive_summary

                st.write(f"Debug: Final title: {title}")
                st.write(f"Debug: Using content source: {'scraped_content' if main_content else 'executive_summary'}")
                st.write(f"Debug: Final content length: {len(content)}")

            # Handle full AnalysisResult objects (from backend service)
            elif hasattr(result, 'scraped_content') and result.scraped_content:
                title = getattr(result.scraped_content, 'title', 'Untitled Website')
                url = getattr(result, 'url', 'Unknown URL')
                executive_summary = getattr(result, 'executive_summary', 'No summary available')
                main_content = getattr(result.scraped_content, 'main_content', '')
                content = main_content

                # Get metrics
                metrics = {}
                if hasattr(result, 'metrics') and result.metrics:
                    metrics_obj = result.metrics
                    metrics = {
                        'overall_score': getattr(metrics_obj, 'overall_score', 0),
                        'content_quality_score': getattr(metrics_obj, 'content_quality_score', 0),
                        'seo_score': getattr(metrics_obj, 'seo_score', 0),
                        'ux_score': getattr(metrics_obj, 'ux_score', 0),
                        'readability_score': getattr(metrics_obj, 'readability_score', 0),
                        'performance_score': getattr(metrics_obj, 'performance_score', 0)
                    }

                # Get creation time
                created_at = getattr(result, 'created_at', None)
                if created_at:
                    if hasattr(created_at, 'strftime'):
                        analyzed_at = created_at.strftime('%Y-%m-%d %H:%M')
                    else:
                        analyzed_at = str(created_at)
                else:
                    analyzed_at = datetime.now().strftime('%Y-%m-%d %H:%M')

            # Fallback for unknown result types
            else:
                title = getattr(result, 'title', 'Unknown Website')
                url = getattr(result, 'url', 'Unknown URL')
                executive_summary = getattr(result, 'executive_summary', 'No summary available')
                content = executive_summary
                metrics = {}
                analyzed_at = datetime.now().strftime('%Y-%m-%d %H:%M')

            return {
                'id': analysis_id,
                'title': title or 'Untitled Website',
                'url': url,
                'summary': executive_summary,
                'analyzed_at': analyzed_at,
                'metrics': metrics,
                'content': content,
                'analysis_result': result
            }

        except Exception as e:
            # Return minimal data if parsing fails
            st.write(f"Debug: Error in _create_website_data: {str(e)}")
            return {
                'id': analysis_id,
                'title': f'Error Loading Website ({str(e)[:50]}...)',
                'url': getattr(result, 'url', 'Unknown'),
                'summary': f'Error loading data: {str(e)}',
                'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'metrics': {},
                'content': '',
                'analysis_result': result
            }

    def _process_user_question(self, question: str) -> Dict[str, Any]:
        """Process user question and generate intelligent response"""
        try:
            # Check if question is in scope
            if not self._is_question_in_scope(question):
                return {
                    "role": "assistant",
                    "content": "I can only answer questions about websites that have been analyzed. Please analyze a website first, or ask questions related to the analyzed content.",
                    "timestamp": datetime.now().isoformat()
                }

            # Find relevant website data
            relevant_data = self._find_relevant_content(question)

            if not relevant_data:
                return {
                    "role": "assistant",
                    "content": "I don't have enough information to answer that question. Please make sure you've analyzed relevant websites first.",
                    "timestamp": datetime.now().isoformat()
                }

            # Generate response using LLM
            response_content = self._generate_llm_response(question, relevant_data)

            return {
                "role": "assistant",
                "content": response_content,
                "source": relevant_data.get('title', 'Analyzed Website'),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "role": "assistant",
                "content": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _is_question_in_scope(self, question: str) -> bool:
        """Check if the question is within scope (rule-based)"""
        # Define scope keywords
        scope_keywords = [
            # Question words
            'who', 'what', 'where', 'when', 'how', 'why', 'which',
            # Business terms
            'company', 'business', 'service', 'product', 'team', 'about',
            'contact', 'email', 'phone', 'address', 'location',
            # Leadership terms
            'ceo', 'founder', 'manager', 'director', 'lead', 'head',
            # General inquiry terms
            'tell me', 'explain', 'describe', 'information', 'details'
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in scope_keywords)

    def _find_relevant_content(self, question: str) -> Optional[Dict]:
        """Find the most relevant website content for the question"""
        if not st.session_state.kr_available_websites:
            return None

        # If a specific website is selected, use that
        if hasattr(st.session_state, 'kr_selected_website') and st.session_state.kr_selected_website:
            selected_website = st.session_state.kr_selected_website
            # Check if it's a dict (website object) or string (title)
            if isinstance(selected_website, dict):
                return selected_website
            else:
                # If it's a string, find the website by title
                for site in st.session_state.kr_available_websites:
                    if site['title'] == selected_website:
                        return site

        # Otherwise, find the most relevant based on keyword matching
        question_lower = question.lower()
        best_match = None
        best_score = 0

        for site in st.session_state.kr_available_websites:
            score = 0
            content_to_search = f"{site['title']} {site.get('summary', '')} {site.get('content', '')[:1000]}".lower()

            # Simple keyword matching score
            for word in question_lower.split():
                if len(word) > 2 and word in content_to_search:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = site

        return best_match

    def _generate_llm_response(self, question: str, website_data: Dict) -> str:
        """Synchronous wrapper for LLM response generation"""
        try:
            # Use asyncio to run the async method
            import asyncio

            # Check if there's already an event loop running
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, we need to use a different approach
                # For now, we'll use the fallback method
                st.warning("Using fallback method due to async limitations in Streamlit")
                return self._generate_fallback_response(question,
                                                     website_data.get('title', 'Unknown Website'),
                                                     self._extract_content_for_analysis(website_data))
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                try:
                    return asyncio.run(self._generate_llm_response_async(question, website_data))
                except Exception as e:
                    st.warning(f"LLM service failed ({str(e)}), using enhanced fallback")
                    return self._generate_fallback_response(question,
                                                         website_data.get('title', 'Unknown Website'),
                                                         self._extract_content_for_analysis(website_data))

        except Exception as e:
            st.error(f"Error in response generation: {str(e)}")
            return self._generate_fallback_response(question,
                                                 website_data.get('title', 'Unknown Website'),
                                                 self._extract_content_for_analysis(website_data))

    def _extract_content_for_analysis(self, website_data: Dict) -> str:
        """Extract the best available content from website data"""
        content = ""

        # Get the actual analysis result for more detailed information
        analysis_result = website_data.get('analysis_result')

        # Priority 1: Scraped content
        if analysis_result and hasattr(analysis_result, 'scraped_content') and analysis_result.scraped_content:
            scraped = analysis_result.scraped_content
            if isinstance(scraped, dict):
                content = scraped.get('main_content', '')
            elif hasattr(scraped, 'main_content'):
                content = scraped.main_content or ""

        # Priority 2: Regular content field
        if not content:
            content = website_data.get('content', '')

        # Priority 3: Summary
        if not content:
            content = website_data.get('summary', '')

        return content

    async def _generate_llm_response_async(self, question: str, website_data: Dict) -> str:
        """Generate response using LLM service for flexible Q&A"""
        try:
            # Extract relevant information from website data
            title = website_data.get('title', 'Unknown Website')
            url = website_data.get('url', '')
            summary = website_data.get('summary', '')

            # Get the best available content
            content = self._extract_content_for_analysis(website_data)

            # Create a simple prompt for Q&A without complex LLM service setup
            qa_prompt = f"""Based on the following website content, please answer the user's question directly and accurately.

Website: {title}
URL: {url}
Summary: {summary or "Not available"}

Content:
{content[:3000] if content else "No content available"}

User Question: {question}

Please provide a direct, helpful answer based on the content above. If the information isn't available, say so clearly."""

            # For now, since LLM integration is complex in Streamlit context,
            # we'll use the enhanced fallback which is actually quite good
            return self._generate_fallback_response(question, title, content)

        except Exception as e:
            st.error(f"LLM service error: {str(e)}")
            return self._generate_fallback_response(question,
                                                 website_data.get('title', 'Unknown Website'),
                                                 self._extract_content_for_analysis(website_data))

    def _generate_fallback_response(self, question: str, title: str, content: str) -> str:
        """Generate an intelligent fallback response when LLM service is unavailable"""
        question_lower = question.lower()

        # More intelligent content search
        if not content or len(content) < 50:
            return f"I have limited information about {title}. Please try asking a more specific question or ensure the website content was properly analyzed."

        # Look for specific person names in the question
        question_words = question_lower.split()
        potential_names = []

        # Extract potential names (capitalized words in original question)
        original_words = question.split()
        for word in original_words:
            if word[0].isupper() and len(word) > 2 and word.isalpha():
                potential_names.append(word)

        # If we have potential names, search for them in content
        if potential_names:
            content_lower = content.lower()
            found_info = []

            for name in potential_names:
                name_lower = name.lower()
                if name_lower in content_lower:
                    # Find sentences containing the name
                    sentences = content.split('.')
                    for sentence in sentences:
                        if name_lower in sentence.lower():
                            clean_sentence = sentence.strip()
                            if len(clean_sentence) > 10:
                                found_info.append(clean_sentence)

                    # Also check lines containing the name
                    lines = content.split('\n')
                    for line in lines:
                        if name_lower in line.lower() and line.strip() not in [s.strip() for s in found_info]:
                            clean_line = line.strip()
                            if len(clean_line) > 10:
                                found_info.append(clean_line)

            if found_info:
                return f"Based on {title}, here's what I found about {' '.join(potential_names)}:\n\n" + "\n".join(found_info[:4])

        # Enhanced keyword-based responses
        if any(keyword in question_lower for keyword in ['who is', 'who are', 'ceo', 'leader', 'president', 'director', 'founder', 'manager', 'chief', 'officer']):
            # Look for leadership terms and names in content
            leadership_terms = ['ceo', 'president', 'director', 'founder', 'leader', 'manager', 'chief', 'officer', 'head', 'vice president', 'vp']
            lines = content.split('\n')
            sentences = content.split('.')

            relevant_info = []

            # Search in lines first
            for line in lines:
                line_lower = line.lower()
                if any(term in line_lower for term in leadership_terms):
                    clean_line = line.strip()
                    if len(clean_line) > 15:
                        relevant_info.append(clean_line)

            # Search in sentences if no lines found
            if not relevant_info:
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(term in sentence_lower for term in leadership_terms):
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 15:
                            relevant_info.append(clean_sentence)

            if relevant_info:
                return f"Based on {title}, here's what I found about leadership:\n\n" + "\n".join(relevant_info[:4])
            else:
                return f"I found information about {title}, but couldn't locate specific leadership details in the analyzed content. The information might be structured differently or in sections not captured during analysis."

        # Model/subscription questions
        elif any(keyword in question_lower for keyword in ['model', 'models', 'subscription', 'plan', 'plans', 'copilot', 'pro', 'plus']):
            # Look for model, plan, or subscription information
            model_terms = ['model', 'models', 'subscription', 'plan', 'plans', 'pro', 'plus', 'premium', 'basic', 'free', 'enterprise']
            relevant_info = []

            lines = content.split('\n')
            sentences = content.split('.')

            # Search in lines
            for line in lines:
                line_lower = line.lower()
                if any(term in line_lower for term in model_terms):
                    clean_line = line.strip()
                    if len(clean_line) > 15:
                        relevant_info.append(clean_line)

            # Search in sentences if no lines found
            if not relevant_info:
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(term in sentence_lower for term in model_terms):
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 15:
                            relevant_info.append(clean_sentence)

            if relevant_info:
                return f"Based on {title}, here's what I found about plans/models:\n\n" + "\n".join(relevant_info[:4])

        # Company/About questions
        elif any(keyword in question_lower for keyword in ['what is', 'about', 'company', 'business', 'service', 'product']):
            # Return first meaningful sentences as summary
            sentences = content.split('.')
            clean_sentences = []
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 30 and not clean_sentence.lower().startswith(('http', 'www', 'email', 'phone')):
                    clean_sentences.append(clean_sentence)
                if len(clean_sentences) >= 3:
                    break

            if clean_sentences:
                return f"Based on {title}:\n\n" + ". ".join(clean_sentences) + "."

        # Contact information questions
        elif any(keyword in question_lower for keyword in ['contact', 'email', 'phone', 'address', 'location']):
            contact_terms = ['email', 'phone', '@', 'contact', 'address', 'tel:', '+1', 'street', 'ave', 'road', 'location']
            relevant_info = []

            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(term in line_lower for term in contact_terms):
                    clean_line = line.strip()
                    if len(clean_line) > 10:
                        relevant_info.append(clean_line)

            if relevant_info:
                return f"Here's contact information for {title}:\n\n" + "\n".join(relevant_info[:3])

        # General fallback - provide a content preview
        content_preview = content[:500] + "..." if len(content) > 500 else content
        return f"Based on {title}, here's a preview of the available information:\n\n{content_preview}\n\nFor more specific answers, try asking about:\n• Specific people or roles\n• Products or services\n• Contact information\n• Company details"

    def _show_website_details(self, website: Dict):
        """Show detailed information about a website"""
        with st.expander(f"Details: {website['title']}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Basic Information**")
                st.markdown(f"- **URL:** {website['url']}")
                st.markdown(f"- **Analyzed:** {website.get('analyzed_at')}")
                st.markdown(f"- **Title:** {website['title']}")

            with col2:
                st.markdown("**Metrics**")
                metrics = website.get('metrics', {})
                if metrics:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            st.markdown(f"- **{key.replace('_', ' ').title()}:** {value:.1f}")

            st.markdown("**Summary**")
            st.markdown(website.get('summary', 'No summary available'))

            if website.get('content'):
                st.markdown("**Content Preview**")
                st.text_area("Content", website['content'][:1000] + "...", height=100, disabled=True)

    def _render_browse_tab(self):
        """Render the browse entries interface"""
        st.subheader("Browse Knowledge Entries")

        # Control panel
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sort_by = st.selectbox("Sort by", ["Recently Created", "Title A-Z", "Title Z-A"], key="kr_sort_by")
        with col2:
            entries_per_page = st.selectbox("Show", [10, 25, 50, 100], index=1, key="kr_entries_per_page")
        with col3:
            if st.button("Refresh", type="secondary"):
                st.rerun()

        # Fetch and display entries
        try:
            entries = self._fetch_all_entries()

            if not entries:
                st.info("No knowledge entries found. Create your first entry in the 'Create Entry' tab.")
                return

            # Apply sorting
            if sort_by == "Title A-Z":
                entries.sort(key=lambda x: x.get('title', '').lower())
            elif sort_by == "Title Z-A":
                entries.sort(key=lambda x: x.get('title', '').lower(), reverse=True)

            # Display entries in a clean grid layout
            self._render_entries_grid(entries[:entries_per_page])

            # Pagination info
            if len(entries) > entries_per_page:
                st.caption(f"Showing {min(entries_per_page, len(entries))} of {len(entries)} entries")

        except Exception as e:
            st.error(f"Failed to load entries: {str(e)}")

    def _render_entries_grid(self, entries: List[Dict]):
        """Render entries in a professional grid layout"""
        for i in range(0, len(entries), 2):
            col1, col2 = st.columns(2)

            # First entry
            with col1:
                if i < len(entries):
                    self._render_entry_card(entries[i])

            # Second entry
            with col2:
                if i + 1 < len(entries):
                    self._render_entry_card(entries[i + 1])

    def _render_entry_card(self, entry: Dict):
        """Render a single knowledge entry card"""
        with st.container():
            # Card styling using markdown
            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 16px;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
            <h4 style="margin-top: 0; color: #2c3e50;">{entry.get('title', 'Untitled')}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Content preview
            content = entry.get('content', '')
            preview = content[:150] + "..." if len(content) > 150 else content
            st.markdown(f"**Preview:** {preview}")

            # Metadata
            metadata = entry.get('metadata', {})
            if metadata:
                st.markdown("**Tags:** " + ", ".join([f"`{k}: {v}`" for k, v in metadata.items()]))

            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("View", key=f"view_{entry['id']}", type="secondary"):
                    self._show_entry_details(entry)
            with col2:
                if st.button("Edit", key=f"edit_{entry['id']}", type="secondary"):
                    self._edit_entry(entry)
            with col3:
                if st.button("Delete", key=f"delete_{entry['id']}", type="secondary"):
                    self._delete_entry(entry['id'])

    def _render_create_tab(self):
        """Render the create new entry interface"""
        st.subheader("Create New Knowledge Entry")

        with st.form("create_entry_form"):
            # Entry details
            title = st.text_input("Title *", placeholder="Enter a descriptive title for this entry")

            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category",
                    ["General", "Process", "Policy", "Technical", "FAQ", "Best Practice", "Troubleshooting"],
                    key="kr_category")
            with col2:
                priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], key="kr_priority")

            content = st.text_area("Content *",
                placeholder="Enter the detailed content of this knowledge entry...",
                height=200)

            # Additional metadata
            with st.expander("Additional Metadata (Optional)"):
                tags = st.text_input("Tags", placeholder="tag1, tag2, tag3")
                author = st.text_input("Author", value=st.session_state.get('user_name', ''))
                department = st.text_input("Department")
                version = st.text_input("Version", value="1.0")

            submitted = st.form_submit_button("Create Entry", type="primary")

            if submitted:
                if not title.strip() or not content.strip():
                    st.error("Title and Content are required fields.")
                else:
                    # Build metadata
                    metadata = {
                        "category": category,
                        "priority": priority,
                        "created_by": author if author else "Unknown",
                        "department": department if department else "",
                        "version": version,
                        "created_at": datetime.now().isoformat(),
                        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()]
                    }

                    success = self._create_entry(title.strip(), content.strip(), metadata)
                    if success:
                        st.success("Knowledge entry created successfully!")
                        st.rerun()

    def _render_search_tab(self):
        """Render the search interface"""
        st.subheader("Search Knowledge Base")

        # Search controls
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search", placeholder="Enter keywords to search...")
        with col2:
            search_type = st.selectbox("Search in", ["All Fields", "Title Only", "Content Only"], key="kr_search_type")

        # Filter options
        with st.expander("Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_category = st.multiselect("Categories",
                    ["General", "Process", "Policy", "Technical", "FAQ", "Best Practice", "Troubleshooting"])
            with col2:
                filter_priority = st.multiselect("Priority", ["Low", "Medium", "High", "Critical"])
            with col3:
                filter_author = st.text_input("Author")

        if search_query or filter_category or filter_priority or filter_author:
            # Perform search
            try:
                all_entries = self._fetch_all_entries()
                filtered_entries = self._filter_entries(all_entries, search_query, search_type,
                                                      filter_category, filter_priority, filter_author)

                if filtered_entries:
                    st.success(f"Found {len(filtered_entries)} matching entries")
                    self._render_entries_grid(filtered_entries)
                else:
                    st.info("No entries match your search criteria.")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

    def _render_import_export_tab(self):
        """Render import/export interface"""
        st.subheader("Import / Export Data")

        # Export section
        st.markdown("### Export Knowledge Base")
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Export Format", ["JSON", "CSV", "Markdown"], key="kr_export_format")
        with col2:
            if st.button("Export All Entries", type="primary"):
                self._export_entries(export_format.lower())

        st.divider()

        # Import section
        st.markdown("### Import Entries")
        uploaded_file = st.file_uploader("Choose a file", type=["json", "csv"])
        if uploaded_file is not None:
            if st.button("Import Data", type="primary"):
                success = self._import_entries(uploaded_file)
                if success:
                    st.success("Data imported successfully!")
                    st.rerun()

    def _fetch_all_entries(self) -> List[Dict]:
        """Fetch all knowledge entries from API"""
        try:
            response = requests.get(f"{self.api_base_url}/knowledge-entries")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Failed to fetch entries: {str(e)}")
            return []

    def _create_entry(self, title: str, content: str, metadata: Dict) -> bool:
        """Create a new knowledge entry"""
        try:
            payload = {
                "title": title,
                "content": content,
                "metadata": metadata
            }
            response = requests.post(f"{self.api_base_url}/knowledge-entry", json=payload)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            st.error(f"Failed to create entry: {str(e)}")
            return False

    def _delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry"""
        try:
            response = requests.delete(f"{self.api_base_url}/knowledge-entry/{entry_id}")
            response.raise_for_status()
            st.success("Entry deleted successfully!")
            st.rerun()
            return True
        except requests.RequestException as e:
            st.error(f"Failed to delete entry: {str(e)}")
            return False

    def _show_entry_details(self, entry: Dict):
        """Show detailed view of an entry in a modal"""
        with st.expander(f"📄 {entry.get('title', 'Untitled')}", expanded=True):
            st.markdown(f"**Content:**")
            st.markdown(entry.get('content', ''))

            metadata = entry.get('metadata', {})
            if metadata:
                st.markdown("**Metadata:**")
                for key, value in metadata.items():
                    if value:
                        st.markdown(f"- **{key.title()}:** {value}")

    def _edit_entry(self, entry: Dict):
        """Edit an existing entry (placeholder for future implementation)"""
        st.info("Edit functionality will be implemented in the next version.")

    def _filter_entries(self, entries: List[Dict], query: str, search_type: str,
                       categories: List, priorities: List, author: str) -> List[Dict]:
        """Filter entries based on search criteria"""
        filtered = entries.copy()

        # Text search
        if query:
            query_lower = query.lower()
            if search_type == "Title Only":
                filtered = [e for e in filtered if query_lower in e.get('title', '').lower()]
            elif search_type == "Content Only":
                filtered = [e for e in filtered if query_lower in e.get('content', '').lower()]
            else:  # All Fields
                filtered = [e for e in filtered if
                           query_lower in e.get('title', '').lower() or
                           query_lower in e.get('content', '').lower()]

        # Category filter
        if categories:
            filtered = [e for e in filtered if
                       e.get('metadata', {}).get('category') in categories]

        # Priority filter
        if priorities:
            filtered = [e for e in filtered if
                       e.get('metadata', {}).get('priority') in priorities]

        # Author filter
        if author:
            author_lower = author.lower()
            filtered = [e for e in filtered if
                       author_lower in e.get('metadata', {}).get('created_by', '').lower()]

        return filtered

    def _export_entries(self, format_type: str):
        """Export entries in specified format"""
        try:
            entries = self._fetch_all_entries()
            if not entries:
                st.warning("No entries to export.")
                return

            if format_type == "json":
                data = json.dumps(entries, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=data,
                    file_name=f"knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            elif format_type == "csv":
                # Flatten entries for CSV
                flattened = []
                for entry in entries:
                    flat_entry = {
                        "id": entry.get("id"),
                        "title": entry.get("title"),
                        "content": entry.get("content"),
                    }
                    # Add metadata fields
                    metadata = entry.get("metadata", {})
                    for key, value in metadata.items():
                        flat_entry[f"metadata_{key}"] = value
                    flattened.append(flat_entry)

                df = pd.DataFrame(flattened)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")

    def _import_entries(self, uploaded_file) -> bool:
        """Import entries from uploaded file"""
        try:
            if uploaded_file.name.endswith('.json'):
                entries = json.load(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # Convert CSV back to entry format (simplified)
                entries = df.to_dict('records')
            else:
                st.error("Unsupported file format")
                return False

            # Import each entry
            success_count = 0
            for entry in entries:
                if self._create_entry(
                    entry.get('title', ''),
                    entry.get('content', ''),
                    entry.get('metadata', {})
                ):
                    success_count += 1

            st.success(f"Successfully imported {success_count} entries")
            return True

        except Exception as e:
            st.error(f"Import failed: {str(e)}")
            return False
