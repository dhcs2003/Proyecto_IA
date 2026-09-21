# Proyecto_IA
Este repositorio contiene el código fuente de un proyecto diseñado con fines educativos para demostrar la aplicación práctica de la Inteligencia Artificial en el desarrollo de software. El juego ilustra cómo integrar modelos de clasificación de Machine Learning (scikit-learn) dentro de un bucle de ejecución en tiempo real (pygame).

A través de un "Jefe Final Adaptativo", el código fuente documentado ejemplifica tres conceptos fundamentales de la IA aplicada:

Recolección de Telemetría (Extracción de Características): Captura de variables de estado del usuario (ratios de daño recibido y métricas de posicionamiento espacial) durante el gameplay sin afectar los cuadros por segundo (FPS).

Entrenamiento y Clasificación Multiclase: Uso de una red neuronal artificial feedforward (MLPClassifier) optimizada con el solver lbfgs sobre un dataset sintético cerrado para mapear el estilo de juego a un perfil de combate.

Inferencia Dinámica: Ejecución de la predicción del modelo en tiempo de ejecución para inyectar el resultado en la máquina de estados del enemigo, logrando un diseño de dificultad reactiva.

Público Objetivo
Este código está estructurado para estudiantes, desarrolladores junior o entusiastas de la programación que buscan un puente práctico entre los fundamentos teóricos de las redes neuronales y su integración en sistemas interactivos orientados a objetos.
