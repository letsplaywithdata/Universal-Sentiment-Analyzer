mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"yogeshnizzer302@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
