import streamlit as st
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from streamlit.report_thread import add_report_ctx

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

ctx = app.app_context()
ctx.push()
add_report_ctx(ctx)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


@app.route('/')
def streamlit_app():
    return render_template('streamlit.html')


@app.route('/register', methods=['POST'])
def register_user():
    username = request.form['username']
    password = request.form['password']

    # Verifica se o usuário já existe no banco de dados
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'message': 'Usuário já registrado. Por favor, faça login.'}), 400

    # Cria um novo usuário e o salva no banco de dados
    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'Registro realizado com sucesso. Por favor, faça login.'}), 201


def main():
    st.header("PDF AI Chatbot")

    # Restante do seu código Streamlit aqui


if __name__ == '__main__':
    db.create_all()
    main()
