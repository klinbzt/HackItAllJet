import openai
from dash import html, dcc, Input, Output, State, ctx
from dash.exceptions import PreventUpdate

class ChatAdvisor:
    def __init__(self, app, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self._build_layout()
        self._register_callbacks(app)

    def _build_layout(self):
        self.chat_ui = html.Div([
            html.H3("💬 AI Model Advisor"),
            dcc.Textarea(
                id='chat-input',
                placeholder='Describe your ML task (e.g., image classification)...',
                style={'width': '100%', 'height': '100px', 'marginBottom': '10px'}
            ),
            html.Button("Ask Advisor", id='chat-submit-btn', n_clicks=0),
            html.Div(id='chat-response', style={
                'marginTop': '20px',
                'whiteSpace': 'pre-wrap',
                'backgroundColor': '#f9f9f9',
                'padding': '10px',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            })
        ])

    def _register_callbacks(self, app):
        @app.callback(
            Output('chat-response', 'children'),
            Input('chat-submit-btn', 'n_clicks'),
            State('chat-input', 'value'),
            prevent_initial_call=True
        )
        def handle_chat(n_clicks, message):
            if not message:
                raise PreventUpdate
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a machine learning assistant that helps users choose the best ML model architecture for a given task."
                        },
                        {"role": "user", "content": message}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                reply = response.choices[0].message.content.strip()
                return f"🤖 Advisor:\n{reply}"
            except Exception as e:
                return f"❌ Error: {str(e)}"

    def get_component(self):
        return self.chat_ui