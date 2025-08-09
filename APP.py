{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3c52e6cc-eac6-4bed-8b79-4775e5aecbeb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "369768e9-a01f-4a76-8770-a4d44cfb938f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6409b691-a2b0-4833-abec-8a25f4c56723",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "65748537-3a03-43ba-a342-2abab60e398b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "6517ec39-c01b-4d6d-8e4f-de630f4ed0ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "import shap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fe1c251c-f861-475c-8774-7ec861538f68",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "01bc8f80-9ee8-4470-849f-268f2f3290e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = joblib.load('RF.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c6568bdc-8edf-4a0e-8f76-ec419b417d77",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = pd.read_csv('X_test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "87667026-d733-4042-a13c-b2420bbd922e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#定义特征名称，对应数据集中的列名\n",
    "feature_names = [\"BC\",\"YiDC\", \"PDC\", \"Age\", \"Pension\", \"WHtR\", \"CO\", \"BMI\", \"Smoking\", \"SCL\", \"Sleepquality\", \"Pain\", \"Eyesight\", \"Mobilitydifficulty\", \"Hyperlipidemia\", \"Hyperuricemia\",\"FLD\", \"OA\", \"Diabetes\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "60814cd5-664d-4e69-be03-f94cb0ed50de",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-09 11:34:11.371 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\chenran\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Streamlit 用户界面\n",
    "st.title(\"Elderly Hypertension Predictor\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "fb7ce107-6268-4473-8a85-e29b1a98c016",
   "metadata": {},
   "outputs": [],
   "source": [
    "BC = st.selectbox(\"Balanced constitution (BC):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8e6b6c60-b020-4f05-9fd7-93e7d767c738",
   "metadata": {},
   "outputs": [],
   "source": [
    "YiDC = st.selectbox(\"Yin-deficiency constitution (YiDC):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "fe271177-8005-4256-a90b-8d89829edc02",
   "metadata": {},
   "outputs": [],
   "source": [
    "PDC = st.selectbox(\"Phlegm-dampness constitution (PDC):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "edd2033c-1065-475a-a287-6c4551e43aba",
   "metadata": {},
   "outputs": [],
   "source": [
    "Age = st.selectbox(\"Age:\", options=[0, 1,2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e79e831c-44b8-441f-9045-edb7892360df",
   "metadata": {},
   "outputs": [],
   "source": [
    "Pension = st.selectbox(\"Pension:\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "8adf9f7f-3ead-4395-968a-3b7c39460f29",
   "metadata": {},
   "outputs": [],
   "source": [
    "WHtR = st.selectbox(\"Waist-to-height ratio (WHtR):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "e4af421d-8d6e-46b6-8376-cd522ae92698",
   "metadata": {},
   "outputs": [],
   "source": [
    "CO = st.selectbox(\"Central obesity (CO):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "84123d5b-b257-479b-b5a1-d5822b622e0d",
   "metadata": {},
   "outputs": [],
   "source": [
    "BMI = st.selectbox(\"Body Mass Index (BMI):\", options=[0,1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "14649e93-d5ff-4280-b0f9-fa2d33da79a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "Smoking = st.selectbox(\"Smoking:\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "3bba232f-c9d1-476c-870d-66b3c573c997",
   "metadata": {},
   "outputs": [],
   "source": [
    "SCL = st.selectbox(\"Spiritual and cultural life (SCL):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "967198a6-fb0d-48ec-84ec-842a5d7fb7c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "Sleepquality = st.selectbox(\"Sleep quality:\", options=[0,1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "476d81df-6e56-4b0a-b5b8-0c77b9ade83c",
   "metadata": {},
   "outputs": [],
   "source": [
    "Pain = st.selectbox(\"Pain:\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "484abe56-b007-46b5-b9c4-cc1f4b9f0071",
   "metadata": {},
   "outputs": [],
   "source": [
    "Eyesight = st.selectbox(\"Eyesight:\", options=[0, 1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "7fac9796-fa82-47e6-84fc-465f2006e32c",
   "metadata": {},
   "outputs": [],
   "source": [
    "Mobilitydifficulty = st.selectbox(\"Mobility difficulty:\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "7af7dcdc-0ede-4dfc-b6b3-396d779d3e09",
   "metadata": {},
   "outputs": [],
   "source": [
    "Hyperlipidemia = st.selectbox(\"Hyperlipidemia:\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "65622038-a527-45e2-9857-b61e0e817cb5",
   "metadata": {},
   "outputs": [],
   "source": [
    "Hyperuricemia = st.selectbox(\"Hyperuricemia:\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "137cfa83-e0b1-47a6-b067-02c7110e463b",
   "metadata": {},
   "outputs": [],
   "source": [
    "FLD = st.selectbox(\"Fatty liver disease (FLD):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "0950a209-e429-4528-8cff-c5fee259f7a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "OA = st.selectbox(\"Osteoarthritis (OA):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "0d17e452-43fb-40ec-9f57-d89d5c58b8e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "Diabetes = st.selectbox(\"Diabetes:\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "3d4cd781-fc8b-4dec-aa45-9b1e854bc52a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 实现输入数据并进行预测\n",
    "feature_values = [BC,YiDC, PDC, Age, Pension, WHtR, CO, BMI, Smoking, SCL, Sleepquality, Pain, Eyesight, Mobilitydifficulty, Hyperlipidemia, Hyperuricemia,FLD, OA, Diabetes]  # 将用户输入的特征值存入列表\n",
    "features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "cd95de61-f782-400a-b09e-661a850dbc9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 当用户点击 \"Predict\" 按钮时执行以下代码\n",
    "if st.button(\"Predict\"):\n",
    "    # 预测类别（0: 无高血压，1: 有高血压）\n",
    "    predicted_class = model.predict(features)[0]\n",
    "    # 预测类别的概率\n",
    "    predicted_proba = model.predict_proba(features)[0]\n",
    "\n",
    "    # 创建 SHAP 解释器，基于树模型（如随机森林）\n",
    "    explainer_shap = shap.TreeExplainer(model)\n",
    "    # 计算 SHAP 值，用于解释模型的预测\n",
    "    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))\n",
    "\n",
    "    # 显示预测结果\n",
    "    st.write(f\"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)\")\n",
    "    st.write(f\"**Prediction Probabilities:** {predicted_proba}\")\n",
    "\n",
    "    # 根据预测结果生成建议\n",
    "    # 如果预测类别为 1（高风险）\n",
    "    if float(predicted_proba[1])>explainer_shap.expected_value[1]:\n",
    "        probability = predicted_proba[1] * 100\n",
    "        advice = (\n",
    "            f\"According to our model, you have a high risk of hypertension. \"\n",
    "            f\"The model predicts that your probability of having hypertension is {probability:.1f}%. \"\n",
    "            \"It's advised to consult with your healthcare provider for further evaluation and possible intervention.\"\n",
    "        )\n",
    "    # 如果预测类别为 0（低风险）\n",
    "    else:\n",
    "        probability = predicted_proba[0] * 100\n",
    "        advice = (\n",
    "            f\"According to our model, you have a low risk of hypertension. \"\n",
    "            f\"The model predicts that your probability of not having hypertension is {probability:.1f}%. \"\n",
    "            \"However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider.\"\n",
    "        )\n",
    "        st.write(advice)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "f957ebdc-cf43-421f-8fbf-346474193c3e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# SHAP 解释\n",
    "st.subheader(\"SHAP Force Plot Explanation\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "34e73dfd-1621-466d-88b1-587d2cabae14",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 创建 SHAP 解释器，基于树模型（如随机森林）\n",
    "explainer_shap = shap.TreeExplainer(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "289237cd-0e3e-41f9-b1f1-0d438faba9df",
   "metadata": {},
   "outputs": [],
   "source": [
    " # 计算 SHAP 值，用于解释模型的预测\n",
    "shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "54c70314-eb45-48bf-837e-42b1b6b9bf49",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\chenran\\anaconda3\\Lib\\site-packages\\shap\\plots\\_force_matplotlib.py:418: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show()\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 根据预测类别显示 SHAP 强制图\n",
    "# 期望值（基线值）\n",
    "# 解释类别 1（患病）的 SHAP 值\n",
    "# 特征值数据\n",
    "# 使用 Matplotlib 绘图\n",
    "shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)\n",
    "# 期望值（基线值）\n",
    "# 解释类别 0（未患病）的 SHAP 值\n",
    "# 特征值数据\n",
    "# 使用 Matplotlib 绘图 \n",
    "plt.savefig(\"shap_force_plot.png\", bbox_inches='tight', dpi=1200)\n",
    "st.pyplot(plt.gcf(), use_container_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fb61573-df43-442f-89e7-54c8f2634e24",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
