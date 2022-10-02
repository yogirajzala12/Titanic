{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "6caecc61",
      "metadata": {
        "id": "6caecc61"
      },
      "source": [
        "# EDA\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 233,
      "id": "260cc97e",
      "metadata": {
        "id": "260cc97e"
      },
      "outputs": [],
      "source": [
        "# Importing the required libraries\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "\n",
        "# load the training and testing data files\n",
        "train_df = pd.read_csv('train.csv')\n",
        "test_df = pd.read_csv('test.csv')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# check to see if the data files have any missing values\n",
        "# RangeIndex: Gives dimension of training set\n",
        "# Also tells us how many non-NA values for each feature\n",
        "train_df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t7Z0QqbGgkG9",
        "outputId": "4ae21931-6d0d-4173-8453-8d31f7ef3b70"
      },
      "id": "t7Z0QqbGgkG9",
      "execution_count": 234,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 12 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  891 non-null    int64  \n",
            " 1   Survived     891 non-null    int64  \n",
            " 2   Pclass       891 non-null    int64  \n",
            " 3   Name         891 non-null    object \n",
            " 4   Sex          891 non-null    object \n",
            " 5   Age          714 non-null    float64\n",
            " 6   SibSp        891 non-null    int64  \n",
            " 7   Parch        891 non-null    int64  \n",
            " 8   Ticket       891 non-null    object \n",
            " 9   Fare         891 non-null    float64\n",
            " 10  Cabin        204 non-null    object \n",
            " 11  Embarked     889 non-null    object \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 83.7+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Description of the features**\n",
        "\n",
        "---\n",
        "\n",
        "PassengerId: Unique ID of a passenger\n",
        "\n",
        "Survived:    0 - Not survived and 1 -  survived\n",
        "\n",
        "Pclass:    Ticket class of passengers. It acts as a proxy for socio-economic status (SES). Pclass value is 1 for upper, 2 for middle and 3 for lower class.\n",
        "\n",
        "Sex:    Sex     \n",
        "Age:    Age (in years). It is fractional if less than 1. If the age is estimated, is it in the form of xx.5    \n",
        "SibSp:    Number of siblings/spouse aboard the Titanic     \n",
        "Parch:    Number of parents / children aboard the Titanic     \n",
        "Ticket:    Ticket number     \n",
        "Fare:    Passenger fare     \n",
        "Cabin:    Cabin number     \n",
        "Embarked:   Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)"
      ],
      "metadata": {
        "id": "8rlQQuMTkLxT"
      },
      "id": "8rlQQuMTkLxT"
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.describe(include= 'all')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 394
        },
        "id": "89vuZM2rja2B",
        "outputId": "7814fec7-eb25-4fd2-d6c3-d2349caff047"
      },
      "id": "89vuZM2rja2B",
      "execution_count": 235,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        PassengerId    Survived      Pclass                     Name   Sex  \\\n",
              "count    891.000000  891.000000  891.000000                      891   891   \n",
              "unique          NaN         NaN         NaN                      891     2   \n",
              "top             NaN         NaN         NaN  Braund, Mr. Owen Harris  male   \n",
              "freq            NaN         NaN         NaN                        1   577   \n",
              "mean     446.000000    0.383838    2.308642                      NaN   NaN   \n",
              "std      257.353842    0.486592    0.836071                      NaN   NaN   \n",
              "min        1.000000    0.000000    1.000000                      NaN   NaN   \n",
              "25%      223.500000    0.000000    2.000000                      NaN   NaN   \n",
              "50%      446.000000    0.000000    3.000000                      NaN   NaN   \n",
              "75%      668.500000    1.000000    3.000000                      NaN   NaN   \n",
              "max      891.000000    1.000000    3.000000                      NaN   NaN   \n",
              "\n",
              "               Age       SibSp       Parch  Ticket        Fare    Cabin  \\\n",
              "count   714.000000  891.000000  891.000000     891  891.000000      204   \n",
              "unique         NaN         NaN         NaN     681         NaN      147   \n",
              "top            NaN         NaN         NaN  347082         NaN  B96 B98   \n",
              "freq           NaN         NaN         NaN       7         NaN        4   \n",
              "mean     29.699118    0.523008    0.381594     NaN   32.204208      NaN   \n",
              "std      14.526497    1.102743    0.806057     NaN   49.693429      NaN   \n",
              "min       0.420000    0.000000    0.000000     NaN    0.000000      NaN   \n",
              "25%      20.125000    0.000000    0.000000     NaN    7.910400      NaN   \n",
              "50%      28.000000    0.000000    0.000000     NaN   14.454200      NaN   \n",
              "75%      38.000000    1.000000    0.000000     NaN   31.000000      NaN   \n",
              "max      80.000000    8.000000    6.000000     NaN  512.329200      NaN   \n",
              "\n",
              "       Embarked  \n",
              "count       889  \n",
              "unique        3  \n",
              "top           S  \n",
              "freq        644  \n",
              "mean        NaN  \n",
              "std         NaN  \n",
              "min         NaN  \n",
              "25%         NaN  \n",
              "50%         NaN  \n",
              "75%         NaN  \n",
              "max         NaN  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-10f04aa6-bb93-46fc-ac03-beab11724a36\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891</td>\n",
              "      <td>891</td>\n",
              "      <td>714.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>204</td>\n",
              "      <td>889</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>unique</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>891</td>\n",
              "      <td>2</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>681</td>\n",
              "      <td>NaN</td>\n",
              "      <td>147</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>top</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>347082</td>\n",
              "      <td>NaN</td>\n",
              "      <td>B96 B98</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>freq</th>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1</td>\n",
              "      <td>577</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>7</td>\n",
              "      <td>NaN</td>\n",
              "      <td>4</td>\n",
              "      <td>644</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.383838</td>\n",
              "      <td>2.308642</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>29.699118</td>\n",
              "      <td>0.523008</td>\n",
              "      <td>0.381594</td>\n",
              "      <td>NaN</td>\n",
              "      <td>32.204208</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>257.353842</td>\n",
              "      <td>0.486592</td>\n",
              "      <td>0.836071</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>14.526497</td>\n",
              "      <td>1.102743</td>\n",
              "      <td>0.806057</td>\n",
              "      <td>NaN</td>\n",
              "      <td>49.693429</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.420000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>223.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>20.125000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>7.910400</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>446.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>14.454200</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>668.500000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>38.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>31.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>512.329200</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-10f04aa6-bb93-46fc-ac03-beab11724a36')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-10f04aa6-bb93-46fc-ac03-beab11724a36 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-10f04aa6-bb93-46fc-ac03-beab11724a36');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 235
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "total = train_df.isnull().sum().sort_values(ascending=False)\n",
        "percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)\n",
        "\n",
        "missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])\n",
        "missing_data.head(5)\n",
        "\n",
        "# Cabin has high number of missing data hence dropping this feature is more logical than imputation \n",
        "# Age and Embarked still has acceptable number of missing entries and hence imputation can be performed here"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "iPO_i9Uj3NyN",
        "outputId": "91824017-8795-405d-8239-5f425893dfad"
      },
      "id": "iPO_i9Uj3NyN",
      "execution_count": 236,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "             Total          %\n",
              "Cabin          687  77.104377\n",
              "Age            177  19.865320\n",
              "Embarked         2   0.224467\n",
              "PassengerId      0   0.000000\n",
              "Survived         0   0.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2dd3d585-3d58-4a64-9f8b-47a0867b739e\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Total</th>\n",
              "      <th>%</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Cabin</th>\n",
              "      <td>687</td>\n",
              "      <td>77.104377</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Age</th>\n",
              "      <td>177</td>\n",
              "      <td>19.865320</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Embarked</th>\n",
              "      <td>2</td>\n",
              "      <td>0.224467</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>PassengerId</th>\n",
              "      <td>0</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Survived</th>\n",
              "      <td>0</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2dd3d585-3d58-4a64-9f8b-47a0867b739e')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2dd3d585-3d58-4a64-9f8b-47a0867b739e button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2dd3d585-3d58-4a64-9f8b-47a0867b739e');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 236
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 237,
      "id": "3394943c",
      "metadata": {
        "id": "3394943c",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "03df843a-eb47-44c4-b9e3-3b627442b4dc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     PassengerId  Survived  Pclass                                       Name  \\\n",
            "61            62         1       1                        Icard, Miss. Amelie   \n",
            "829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)   \n",
            "\n",
            "        Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked  \n",
            "61   female  38.0      0      0  113572  80.0   B28      NaN  \n",
            "829  female  62.0      0      0  113572  80.0   B28      NaN  \n"
          ]
        }
      ],
      "source": [
        "# We’ll need to fill the two missing values for Embarked. \n",
        "# Taking a quick look at the two passengers that don’t have values for Embarked\n",
        "# Inner bracket gives boolean output and the outer train_df gives rows having null values as the output\n",
        "\n",
        "print (train_df[train_df.Embarked.isnull()])"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Imputation in 'Embarked' feature - Approach 1"
      ],
      "metadata": {
        "id": "l9xh9p8r1clM"
      },
      "id": "l9xh9p8r1clM"
    },
    {
      "cell_type": "code",
      "execution_count": 238,
      "id": "85be6cd3",
      "metadata": {
        "id": "85be6cd3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "82c5d76e-5b03-4def-d23c-3706ed6081c3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Embarked        C   Q    S\n",
            "Sex    Pclass             \n",
            "female 1       43   1   48\n",
            "       2        7   2   67\n",
            "       3       23  33   88\n",
            "male   1       42   1   79\n",
            "       2       10   1   97\n",
            "       3       43  39  265\n"
          ]
        }
      ],
      "source": [
        "# pivot table shows a breakdown by Sex, Pclass, Embarked, and shows the number of people from each subset that survived and embarked at a specific port\n",
        "# This approach suggests that the imputation should be done with 'C' values as it is the most probable values for female passengers of class 1\n",
        "print (train_df.pivot_table(values='Survived', index=['Sex', 'Pclass'], \n",
        "                     columns=['Embarked'], aggfunc='count'))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Imputation in 'Embarked' feature - Approach 2"
      ],
      "metadata": {
        "id": "N9oz6qpq1wjW"
      },
      "id": "N9oz6qpq1wjW"
    },
    {
      "cell_type": "code",
      "source": [
        "bins = range(0,100,10)\n",
        "df = train_df.copy()\n",
        "df['Age1'] = pd.cut(df['Age'], bins)\n",
        "\n",
        "#First filter the df of females who survived\n",
        "#Create a pivot table on basis of age bins created before with column as embarkment\n",
        "#This shows the missing data should be from S class \n",
        "# We will use 'S' for imputation as this approach is more logical and reliable\n",
        "\n",
        "\n",
        "df1 = df[(df.Survived == 1) & (df.Sex == \"female\")]\n",
        "print (df1.pivot_table(values='Survived', index=['Age1','Pclass'], \n",
        "                     columns=['Embarked'], aggfunc=['count']))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mqhJkFTQ1A44",
        "outputId": "81691b7f-a02b-45c9-f11d-f3e795e80ad5"
      },
      "id": "mqhJkFTQ1A44",
      "execution_count": 239,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                count       \n",
            "Embarked            C  Q   S\n",
            "Age1     Pclass             \n",
            "(0, 10]  1          0  0   0\n",
            "         2          1  0   7\n",
            "         3          5  0   6\n",
            "(10, 20] 1          5  0   8\n",
            "         2          2  0   6\n",
            "         3          4  4   5\n",
            "(20, 30] 1         10  0  10\n",
            "         2          4  1  20\n",
            "         3          2  1  13\n",
            "(30, 40] 1          9  1  13\n",
            "         2          0  0  16\n",
            "         3          0  0   6\n",
            "(40, 50] 1          7  0   5\n",
            "         2          0  0   9\n",
            "         3          0  0   0\n",
            "(50, 60] 1          6  0   5\n",
            "         2          0  0   2\n",
            "         3          0  0   0\n",
            "(60, 70] 1          0  0   1\n",
            "         2          0  0   0\n",
            "         3          0  0   1\n",
            "(70, 80] 1          0  0   0\n",
            "         2          0  0   0\n",
            "         3          0  0   0\n",
            "(80, 90] 1          0  0   0\n",
            "         2          0  0   0\n",
            "         3          0  0   0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 240,
      "id": "9160b1e3",
      "metadata": {
        "id": "9160b1e3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "98f66d86-16f7-4974-fcce-6ba72830c6d1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/pandas/core/indexing.py:1732: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  self._setitem_single_block(indexer, value, name)\n"
          ]
        }
      ],
      "source": [
        "# Finally, imputation of missing values by 'S'\n",
        "(train_df.Embarked.iloc[61]) = 'S'\n",
        "(train_df.Embarked.iloc[829]) = 'S'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 241,
      "id": "7ff94f79",
      "metadata": {
        "id": "7ff94f79"
      },
      "outputs": [],
      "source": [
        "le_Sex = LabelEncoder()\n",
        "train_df.Sex = le_Sex.fit_transform(train_df.Sex)\n",
        "test_df.Sex = le_Sex.transform(test_df.Sex)\n",
        "\n",
        "le_Embarked = LabelEncoder()\n",
        "train_df.Embarked = le_Embarked.fit_transform(train_df.Embarked)\n",
        "test_df.Embarked = le_Embarked.transform(test_df.Embarked)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Dealing with missing entries of 'Age' feature"
      ],
      "metadata": {
        "id": "WyIbM_tKKz1s"
      },
      "id": "WyIbM_tKKz1s"
    },
    {
      "cell_type": "code",
      "source": [
        "# We can use the classical method such as imputation with mean \n",
        "# Another approach is generating a list of random numbers (with size = df['Age'].isnull() and values mean + std or mean - std) and filling NaN values with this list\n",
        "\n",
        "# We can do even better by using the P_class feature!\n",
        "# Pclass does not contain any missing data entries also we may see a relation of the passenger class with regards to the age of passengers\n",
        "# As seen here the young people are more likely to travel in class 3 (cheapest)\n",
        "\n",
        "sns.boxplot(x='Pclass',y='Age',data=train_df)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "yGEIt3POEwj2",
        "outputId": "47c12c9d-fc18-453a-bc67-13634e063ab2"
      },
      "id": "yGEIt3POEwj2",
      "execution_count": 242,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.groupby(by='Pclass').mean()['Age']\n",
        "# These values can be imputed wherever the age is missing"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EagWKfYZMOdx",
        "outputId": "2fefb827-4098-4384-e923-4f34013b8b9c"
      },
      "id": "EagWKfYZMOdx",
      "execution_count": 243,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Pclass\n",
              "1    38.233441\n",
              "2    29.877630\n",
              "3    25.140620\n",
              "Name: Age, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 243
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Now, the aim is to fill in the Above values wherever age is missing. \n",
        "# Function can be made as shown here for this purpose\n",
        "\n",
        "# ac = [['Age','Pclass']]\n",
        "def impute_age(ac):\n",
        "    Age = ac[0]\n",
        "    Pclass = ac[0]\n",
        "    \n",
        "    if pd.isnull(Age):\n",
        "        \n",
        "        if Pclass == 1:\n",
        "            return 38\n",
        "        elif Pclass == 2:\n",
        "            return 29\n",
        "        else:\n",
        "            return 25\n",
        "    \n",
        "    else:\n",
        "        return Age"
      ],
      "metadata": {
        "id": "iVINHbzAMefD"
      },
      "id": "iVINHbzAMefD",
      "execution_count": 244,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data = [train_df,test_df]\n",
        "for dataset in data:\n",
        "  dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis=1).astype(int) "
      ],
      "metadata": {
        "id": "pqjXXJxZMy-R"
      },
      "id": "pqjXXJxZMy-R",
      "execution_count": 245,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sns.heatmap(train_df.isnull())\n",
        "# Now the only feature having missing values is Cabin but we are gonna drop this feature as it has 77% missing values!!\n",
        "train_df = train_df.drop('Cabin',axis=1)\n",
        "test_df = test_df.drop('Cabin',axis=1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 321
        },
        "id": "AE-SpZmxNSBH",
        "outputId": "917a4928-0513-472c-b160-4390c4b3799c"
      },
      "id": "AE-SpZmxNSBH",
      "execution_count": 246,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWYAAAEwCAYAAACE8dv8AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXxM1//48dckEfsaNbagtlJRSymK0GiExJqlQVFFtdZqixa1l+iiaq+llpSqShGRSBAqqoK2SGl9WkuISkITQnaZub8/8st8xZJMMndkRt7Pz2MeH3PnzrlnUvN2cu77fY5GURQFIYQQFsOmqDsghBAiNwnMQghhYSQwCyGEhZHALIQQFkYCsxBCWBgJzEIIYWHMFpgjIiJwc3PD1dWVNWvWmOsyQgjx1DFLYNbpdMydO5d169YRHBzMnj17uHDhgjkuJYQQRWrq1Kl06NCBXr16PfJ1RVH45JNPcHV1pXfv3pw7dy7fNs0SmKOioqhbty6Ojo7Y29vj4eFBeHi4OS4lhBBFytPTk3Xr1j329YiICKKjo9m3bx/z5s1j9uzZ+bZplsAcHx9P9erVDc+1Wi3x8fHmuJQQQhSptm3bUrFixce+Hh4eTr9+/dBoNLRs2ZI7d+5w48aNPNuUm39CCGFGDw5Uq1evnu9A1c4cHdFqtcTFxeXqmFarfXwn7GuZoxtCiHykXT9ilnZL1+xslnYBsjL/NbmNe/9dMvrcHeEn2bZtm+G5r68vvr6+JvchL2YJzM2bNyc6OpqYmBi0Wi3BwcEsWrTIHJcSQoiC0+uMPtXUQPzgQDUuLi7PgSqYaSrDzs6OmTNnMnLkSNzd3enZsyeNGjUyx6WEEKLgFL3xDxO5uLiwa9cuFEXh9OnTlC9fnmrVquX5HrOMmAG6dOlCly5dzNW8EEIUnt70gJvj/fff58SJE9y6dQtnZ2fGjx9PVlYWAAMHDqRLly4cPnwYV1dXSpcuzYIFC/JtU2MJ6zHLHLMQwlhqzDFnXs8/lziHfc1mJl+voMw2YhZCWD5rvPmnCl1WUfcgTyYHZp1Oh5eXF1qtltWrVzNt2jTOnj2Loig8++yz+Pn5UbZsWTX6KoQQ6ijAzb+iYPLNP39/fxo0aGB4Pm3aNHbv3k1QUBA1atRgy5Ytpl5CCCHU9QRv/hWGSYE5Li6On376CW9vb8OxcuXKAdn14enp6ab1TgghzEGvN/5RBEwKzAsWLGDy5MnY2ORuZurUqXTs2JFLly4xZMgQkzoohBBqUxS90Y+iUOjAfOjQIapUqYKTk9NDr/n5+XHkyBEaNGhASEiISR0UQgjVPa0j5t9//52DBw/i4uLC+++/T2RkJJMmTTK8bmtri4eHB/v27VOlo0IIoRrdPeMfRaDQWRkffPABH3zwAQDHjx9n/fr1fP7551y5coW6deuiKAoHDx6kfv36qnVWCCFUUURTFMZSNY9ZURQ+/PBDUlJSUBSF5557jjlz5qh5CSGEMF0RTVEYSyr/hBBWRY3Kv4yz+40+t6STq8nXKyip/BNCFD8WPmI2KTBv3LiR7du3o9FoaNy4MX5+ftjb2/PVV18RGhqKjY0NAwcOZOjQoWr1VwihouJakq3oi+amnrEKHZjj4+Px9/cnJCSEUqVK8e677xIcHIyiKMTGxrJ3715sbGxISEhQs79CCGE6Cx8xm1RgotPpSE9PJysri/T0dKpVq8bWrVsZO3asoejEwcFBlY4KIYRqntaSbK1Wy/Dhw3nllVfo1KkT5cqVo1OnTsTExBASEoKnpycjR44kOjpaxe4KIYQK9DrjH0Wg0IE5KSmJ8PBwwsPDOXLkCGlpaQQGBpKZmUnJkiXZsWMHr732GtOmTVOzv0IIYbqndcT8yy+/ULt2bapUqUKJEiXo3r07p06dQqvV4uqanV7i6urK//73P9U6K4QQqrDwkuxC3/yrWbMmZ86cIS0tjVKlSnHs2DGcnJwoV64cx48fx9HRkRMnTlCvXj0VuyuEECp4WhfKb9GiBW5ubvTv3x87OzuaNm2Kr68v6enpTJo0iU2bNlGmTBnmz5+vZn+FEMJ0Fp6VIZV/QhRz5shlNmcesxqVf2kRG40+t7TzMJOvV1BS+SdEMWauAhOLZ+EjZgnMQojix9pXl5s6dSo//fQTDg4O7NmzB4BPP/2UQ4cOUaJECerUqYOfnx8VKlQgMzOTWbNmcfbsWTQaDdOnT6ddu3Zm/xBCCFEgFj5izjddztPTk3Xr1uU61rFjR/bs2UNQUBD16tVj9erVAGzfvh2AoKAgNmzYwKefforewn8AQohiSJdl/KMI5Dtibtu2LdeuXct1rFOnToY/t2zZktDQUAAuXLhgGCE7ODhQvnx5zp49ywsvvKBmn4UQKrH0xYbMxtqnMvLz448/0rNnTwCaNGnCwYMH6dWrF7GxsZw7d47Y2FgJzEJYqOK6upylT2WYFJhXrVqFra0tffr0AcDLy4uLFy/i5eVFzZo1adWqFba2tqp0VAghVPO0BuYdO3bw008/sXHjRjQaTXZjdna51sYYMGCAVP4JISzP0ziVERERwbp169i8eTOlS5c2HE9LS0NRFMqUKcPRo0extbWlYcOGqnVWCKEui59yMBdrL8l+//33OXHiBLdu3cLZ2Znx48ezZs0aMjMzefPNN4Hs8uy5c+eSkJDAiBEjsLGxQavV8tlnn5n9AwghCk/mmC2TlGQLIayKKiXZOxYYfW5pzye/dLFU/glRjMmI2TIVqvJv4sSJXL58GYC7d+9Svnx5AgMDOXr0KIsWLeLevXuUKFGCyZMn06FDB/N+AiGEKChrD8yenp4MHjyYDz/80HDsq6++Mvx54cKFlCtXDoDKlSuzatUqtFotf//9NyNGjODIkWK6SIoQwnIV/QxungpV+ZdDURT27t3Lpk2bAHj++ecNrzVq1IiMjAwyMzOxt7dXqbtCCKGCLMvOyjBpl+xff/0VBweHR+Yqh4WF8fzzz0tQFkJYHhX3/IuIiMDNzQ1XV1fWrFnz0OvXr19nyJAh9OvXj969e3P48OF82zTp5t+ePXvo1avXQ8f/+ecfvvjiC9avX29K80IIYR4qzTHrdDrmzp3Lhg0b0Gq1eHt74+Likqt+Y9WqVfTs2ZNBgwZx4cIFRo0axcGDB/Nst9Aj5qysLPbv34+7u3uu43FxcYwbN45PP/2UOnXqFLZ5IYQwH0Ux/pGHqKgo6tati6OjI/b29nh4eBAeHp7rHI1GQ3JyMpCdLFGtWrV8u1foEfMvv/xC/fr1qV69uuHYnTt3GDVqFB988AEvvvhiYZsWQgjzKsCIedu2bWzbts3w3NfXF19fXwDi4+NzxUCtVktUVFSu948bN44RI0awefNm0tLS2LBhQ77XLFTln4+PDyEhIXh4eOQ6d/PmzVy9epUVK1awYsUKANavX4+Dg0O+HRFCiCemAIH5/kBcGMHBwfTv35/hw4dz6tQppkyZwp49e7CxefyERb6B+csvv3zk8YULFz50bMyYMYwZM6YAXRZCPI3Srh+x6CITRadTpR2tVktcXJzheXx8PFqtNtc5AQEBhs1GWrVqRUZGBrdu3cpzwCqVf0IUY5YcPM1KpZt/zZs3Jzo6mpiYGLRaLcHBwSxatCjXOTVq1ODYsWN4enpy8eJFMjIyqFKlSp7tSmAWohgrtiXZKi37aWdnx8yZMxk5ciQ6nQ4vLy8aNWrEkiVLcHJyolu3bnz00Ud8/PHHhiWSFy5caFgq+XHyXcQoNjaWKVOmkJCQgEaj4bXXXuONN95g7969LF++nIsXL7J9+3aaN28OwLVr13B3d+fZZ58F/m/luTw/nCxiJESRsMbArMYiRqkrxhl9bpmxy02+XkHlO2K2tbXlo48+olmzZiQnJ+Pl5UXHjh1p3Lgxy5YtY9asWQ+9p06dOgQGBpqlw0II9Vj8yNZcrH2tjGrVqhny7sqVK0f9+vWJj4+nY8eOZu+cEEKYhUo3/8ylQHPM165d46+//qJFixb5ntevXz/KlSvHxIkTadOmjUmdFEKYhzVOZajC2kfMOVJSUpgwYQLTpk0zrCb3KNWqVePQoUNUrlyZs2fPMnbsWIKDg/N8jxBCPFF6y15dzqiS7Hv37jFhwgR69+5N9+7d8zzX3t6eypUrA+Dk5ESdOnUMazcLIYRFUHERI3PINzArisL06dOpX7++YY+/vCQmJqL7//M3MTExREdH4+joaHpPhRBCLXrF+EcRyHcq47fffiMwMJDGjRvTt29fILtMOzMzk3nz5pGYmMjbb79N06ZN+eabbzh58iRLly7Fzs4OGxsb5syZQ6VKlcz+QYQQBWfxc8Fmolj4HLNsxipEMWaNN//UyGNO+WSw0eeW/XizydcrKKn8E0IUPxZ+8y/fwPy4yr9ly5bxww8/GGq+33//fbp06cK9e/f4+OOP+fPPP8nKyqJfv368/fbbZv8gQghhNAufyih05R/AsGHDGDFiRK7zQ0NDyczMJCgoiLS0NDw8PPDw8KB27drm+QRCCFFQ1j5iflzl3+NoNBrS0tLIysoiPT2dEiVKSA6zEMKyFFEanLEKtLXUg5V/W7ZsoXfv3kydOpWkpCQA3NzcKF26NJ06deKVV15h+PDhkpUhhLAsFp4uZ3RgfrDyb+DAgezfv5/AwECqVatmWDg/KioKGxsbjhw5Qnh4OOvXrycmJsZsH0AIIQpKydIZ/SgKRmVlPKryr2rVqobXfXx8eOedd4DsnbM7d+5MiRIlcHBwoHXr1vzxxx9SZCKEBSquecxWP8f8uMq/GzduGOaeDxw4QKNGjYDs1fqPHz9Ov379SE1N5cyZM7zxxhtm6r4QwhTWmMesCgufYy505d+ePXs4f/48ALVq1TIshv/6668zdepUPDw8UBQFT09PmjRpYsaPIIQQBWThI2ap/BNCWBU1Kv/uTuxt9Lnlvwoy+XoFJZV/QhRjxXYqo4hu6hlLArMQovix8KmMfNPlMjIy8Pb2pk+fPnh4eLB06VIge0lPHx8fXF1dmThxIpmZmQCcPHmS/v378/zzzxMaGmre3gshRGFYeB5zviNme3t7Nm3aRNmyZbl37x6DBg3C2dmZDRs2MGzYMDw8PJg5cyYBAQEMGjSIGjVq4Ofnx/r1659E/4UQJrD4KQczsYBba3nKNzBrNBrKli0LQFZWFllZWWg0GiIjI1m0aBEA/fv3Z/ny5QwaNMiwJoaNTYGKCoUQRaDYzjFb+FSGUXPMOp0OT09Prl69yqBBg3B0dKRChQrY2WW/vXr16nmunyGEsEwWH0DN5WkIzLa2tgQGBnLnzh3Gjh3LpUuXzN0vIYQwGyXLygtM7lehQgXatWvH6dOnuXPnDllZWdjZ2REXF4dWqzVXH4UQQl2WHZfzz8pITEzkzp07AKSnp/PLL7/QoEED2rVrR1hYGAA7d+7ExcXFvD0VQgiVKHrF6EdRyLfy7/z583z00UfodDoURaFHjx6MGzeOmJgY3nvvPZKSkmjatClffPEF9vb2REVFMW7cOO7cuUPJkiWpWrUqwcHBeXZCKv+EKBrmuvkH5pu/VqPy7/bAV4w+t9LWQyZfr6CkJFuIYswaszJUCcy+BQjM2558YJbKPyFEsVNUUxTGKnTlX45PPvmEVq1aPfS+sLAwnnvuOf744w/1eiuEECpQshSjH0Wh0JV/LVu25I8//jBsKXW/5ORk/P39DVtQCSEsU/HNYy7qDuQt3xHz4yr/dDodn332GZMnT37oPUuWLOGtt96iZMmS6vdYCCFMpOiNfxQFo+qmdTodffv25eWXX+bll1+mRYsWbN68mW7duhl2Mclx7tw54uLi6Nq1qzn6K4QQptMX4FEEClX5d/LkSUJDQ/n2229znafX61m4cCF+fn5m6awQQl3WmJWhBjVHwhEREcyfPx+9Xo+Pjw+jRo166JyQkBCWL1+ORqOhSZMmhnWGHqdQlX/Hjx/n6tWrho1Z09LScHV1ZceOHfz9998MHToUgJs3bzJ69GhWrVpF8+bNC3IpIYQwGyVLnXZ0Oh1z585lw4YNaLVavL29cXFxoWHDhoZzoqOjWbNmDVu3bqVixYokJCTk226+gTkxMRE7OzsqVKhgqPx76623OHr0qOGcVq1asX//fgCOHz9uOD5kyBCmTJkiQVkIYVHUGjFHRUVRt25dHB0dAfDw8CA8PDxXYP7hhx94/fXXqVixIgAODg75tptvYL5x48ZDlX+vvGJ8crYQovhJu37Eoqcz1ArM8fHxVK9e3fBcq9USFRWV65zo6GgABgwYgF6vZ9y4cTg7O+fZbr6BuUmTJuzatSvPc06dOvXI4w/OQQshigdLDsoAKBqjT922bRvbtm0zPPf19cXX19fo9+t0Oq5cucK3335LXFwcgwcPJigoiAoVKjz2PVL5J4QodgoyYs4rEGu1WuLi4gzP4+PjH1ppU6vV0qJFC0qUKIGjoyP16tUjOjqaF1544bHXlMAsRDFm8SNbM1H0xo+Y89K8eXOio6OJiYlBq9USHBz8UMbFq6++SnBwMF5eXiQmJhIdHW2Yk36cfANzRkYGr7/+OpmZmeh0Otzc3JgwYQLHjh3js88+Q6/XU6ZMGRYuXEjdunVZsGCB4QZgeno6CQkJ/PrrryZ8dCGEuRTXdDm9Tp3AbGdnx8yZMxk5ciQ6nQ4vLy8aNWrEkiVLcHJyolu3bnTu3JmjR4/i7u6Ora0tU6ZMoXLlynm2m+/qcoqikJqamqske/r06Xz44YesXLmSBg0asGXLFv744w8WLlyY673ffvstf/75Z755zbK6nBDCWGqsLnetnfHrx9c+ftDk6xVUoTdjhew1MXL+/8EKQIDg4GDGjx+vZn+FECoqriNmtaYyzKVQm7G2aNGC+fPnM2rUKEqWLEm5cuX44Ycfcr3n33//5dq1a7Rv394sHRdCiMIq+lXo82bUWhk5JdmHDx8mKiqKv//+m40bN7JmzRoiIiLw9PR8aLoiODgYNzc3bG1tzdJxIYQoLEWvMfpRFApVkh0REcH58+cNy3q6u7szcuTIXOeGhIQwc+ZM9XoqhFCdpU85mItaN//MpdAl2Xfv3uXy5cs8++yzHD16lAYNGhjec/HiRe7cufPIBfSFEJbDGvf8U4PVzzE/riT7k08+YcKECWg0GipWrMiCBQsM7wkJCcHd3d1wk1AIUbxYclAGUApQ+VcUZDNWIYoxa8zKUCNd7sLzbkaf2/DPMJOvV1BS+SeEKHb0Fj5iNjow51S1aLVaVq9ezbRp0zh79iyKovDss8/i5+dH2bJlyczMZMqUKZw7d45KlSqxePFiateubc7PIIQQBWLpUxlGpcsB+Pv757rBN23aNHbv3k1QUBA1atRgy5YtAGzfvp0KFSqwf/9+hg0bxhdffKF+r4UQwgR6ncboR1EwKjDHxcXx008/4e3tbThWrlw5ILtkOz093XD84MGD9O/fHwA3NzeOHTuGBUxjCyGEgaXnMRsVmBcsWMDkyZOxscl9+tSpU+nYsSOXLl1iyJAhQPaydzVq1ACyF/goX748t27dUrnbQghReHpFY/SjKOQ7x3zo0CGqVKmCk5NTrm2jAPz8/NDpdMybN4+QkBC8vLzM1lEhhPosPa3NXKx+jvn333/n4MGDuLi48P777xMZGcmkSZMMr9va2uLh4cG+ffuA7EWhY2NjgexFj+7evZvvEndCCPEkKYrxj6KQ74j5gw8+4IMPPgCyN1pdv349n3/+OVeuXKFu3booisLBgwepX78+AC4uLuzcuZNWrVoRFhZG+/btpdBECAtmjbnMpnpq0uXupygKH374ISkpKSiKwnPPPcecOXMA8Pb2ZvLkybi6ulKxYkUWL16saoeFEOopjkEZQG/hJdlS+SdEMWaNgVmNyr9fa/cz+tw21/LejNocpPJPCFHsWPrNPwnMQohi56mZY36wJHvQoEGkpKQAkJCQwAsvvMDKlSs5cOAAS5YswcbGBltbW6ZNm0abNm3M9gGEEKKginz+Nh9GB+ackuycff6+++47w2vjx4+nW7duAHTo0IFu3bqh0Wg4f/48EydOJDQ0VOVuCyFE4en0Rq9GUSQKXZKdIzk5mcjISF599VUAypYta0iPS0tLk1Q5IYTF0RfgURSMGjHnlGTnTF3c78CBA3To0MGwdgbA/v37WbRoEYmJiaxevVq93gohhAoULHvAaFJJNsCePXvw8fHJdczV1RVXV1dOnjzJkiVL2Lhxo2odFkKox9Lzjc1Fb+GTzPkG5pyS7IiICDIyMkhOTmbSpEl88cUXJCYm8scff7BixYpHvrdt27bExMSQmJhIlSpVVO+8EMI01pjHrAa9tY+YH1WSnbPGclhYGF27dqVkyZKG869cuUKdOnXQaDScO3eOzMxMWStDCAtl6QHUXKx+KiMvISEhvPXWW7mOhYWFERgYiJ2dHaVKlWLx4sVyA1AIC1VcR8w6Cw/MUpIthLAqapRkh2oHGH1uj/jvTb5eQUnlnxCi2CmqNDhjGRWYXVxcKFu2rKGab8eOHezdu5fly5dz8eJFtm/fTvPmzQ3nnz9/nlmzZpGcnIyNjQ0BAQG55qGFEJahuE5lPDVzzJs2bcqVWdG4cWOWLVvGrFmzcp2XlZXF5MmT+fzzz2nSpAm3bt3Czk4G5kJYIksPoOZi4at+Fn4q4/4ds+939OhRnnvuOZo0aQIgGRlCWLDiOmK2+nS5HCNGjECj0eDr64uvr+9jz7t8+TIajYYRI0aQmJiIu7v7Q5kbQgjLYOkB1Fx0Rd2BfBgVmLdu3YpWqyUhIYE333yT+vXr07Zt20eeq9Pp+O233wgICKB06dIMGzYMJycnOnTooGrHhRCisPQWnsJr1CJGWq0WAAcHB1xdXYmKinrsudWrV6dt27ZUqVKF0qVL4+zszLlz59TprRBCqEApwKMo5DtiTk1NRa/XU65cOVJTUzl69Chjxox57PmdOnVi3bp1pKWlUaJECU6ePMmwYcPU7LMQQiXmmmMGy54mUTNdLiIigvnz56PX6/Hx8WHUqFGPPC8sLIwJEyYQEBCQK4vtUfINzAkJCYwdOxbInqbo1asXzs7O7N+/n3nz5pGYmMjbb79N06ZN+eabb6hYsSLDhg3D29sbjUaDs7MzXbt2LfinFUJYLUsOyqBeVoZOp2Pu3Lls2LABrVaLt7c3Li4uNGzYMNd5ycnJ+Pv706JFC6PazTcwOzo6snv37oeO56wg9yh9+/alb9++RnVACCGeNLVKsqOioqhbty6Ojo4AeHh4EB4e/lBgXrJkCW+99RbffPONUe1a9jL+QghhBnqN8Y9t27bh6elpeGzbts3QTnx8PNWrVzc812q1xMfH57rWuXPniIuLK9DMgVR+CCGKnYLMMeeXIpzndfR6Fi5ciJ+fX4HeV+iS7GXLlvHDDz8YqgHff/99unTpQmZmJrNmzeLs2bNoNBqmT59Ou3btCv6JhBDCTNTKttBqtcTFxRmex8fHG7LYAFJSUvj7778ZOnQoADdv3mT06NGsWrUqzxuAhS7JBhg2bBgjRozIdWz79u0ABAUFkZCQwFtvvUVAQAA2NjJrIoSwDGrd/GvevDnR0dHExMSg1WoJDg5m0aJFhtfLly+fa+enIUOGMGXKlHyzMlSPlhcuXDCMkB0cHChfvjxnz55V+zJCCFFoam3Gamdnx8yZMxk5ciTu7u707NmTRo0asWTJEsLDwwvdP5NKsrds2cKuXbtwcnLio48+omLFijRp0oSDBw/Sq1cvYmNjOXfuHLGxsbzwwguF7qQQQqhJp2LhX5cuXejSpUuuY+++++4jz/3222+NarPQJdkDBw5kzJgxaDQalixZYpjg9vLy4uLFi3h5eVGzZk1atWqFra2tUZ0RQognwdLXYy50SXbVqlWxtbXFxsYGHx8f/vjjDyB7aD9t2jQCAwNZtWoVd+/epV69emb7AEIIy2POikI1qDWVYS75BubU1FSSk5MNfz569CiNGjXixo0bhnMOHDhAo0aNAEhLSyM1NRXIXgLU1tb2oWRrIcTTzdIr/6x+rYzHlWRPnjyZ8+fPA1CrVi3mzp1rOH/EiBHY2Nig1Wr57LPPzNh9IYQoOEtfKF82YxWiGLPGhfLV2Ix1cZ3BRp/73tXNJl+voKTyTwhR7Fj6QvlG3fy7c+cOEyZMoEePHvTs2ZNTp05x+/Zt3nzzTbp3786bb75JUlISALt376Z379707t2bAQMGGKY7hBDCUhRkrYyiYFRgnj9/Pp07dyY0NJTAwEAaNGjAmjVr6NChA/v27aNDhw6sWbMGgNq1a7N582aCgoIYPXo0M2bMMOsHEEKIgrL6rIy7d+9y8uRJvL29AbC3t6dChQqEh4fTr18/APr168eBAwcAaN26NRUrVgSgZcuWuerIhRDCElh9Vsa1a9eoUqUKU6dO5fz58zRr1ozp06eTkJBAtWrVAHjmmWdISEh46L0BAQE4Ozur32shhDCBvshCrnHyHTFnZWXx559/MnDgQHbt2kXp0qUN0xY5NBoNmgc2N4yMjCQgIIBJkyap22MhhDCRrgCPopBvYK5evTrVq1c3bInSo0cP/vzzTxwcHAxFJjdu3Mi18tz58+f5+OOPWblyJZUrVzZT14UQpjJXWptU/pkm38D8zDPPUL16dS5dugTAsWPHaNCgAS4uLuzatQuAXbt20a1bNwCuX7/O+PHj+eyzz3j22WfN2HUhhKmsMY9ZDZaelWFUHvOMGTOYNGkS9+7dw9HRET8/P/R6PRMnTiQgIICaNWvy1VdfAbBixQpu377NnDlzAAwL6wshhKWw9DlmqfwTohizxhGzGpV/0+sNMvrc+dHfmXy9gpLKPyFEsWPpy35KYBaiGLP0uWBz0Vn4VIZRgfnOnTt8/PHH/P3332g0GhYsWECrVq0AWL9+PZ9++inHjh2jSpUqHD9+nDFjxlC7dm0AXF1dGTdunPk+gRCi0KxxKkMNT8WIOacke+nSpWRmZpKeng5AbGwsR48epWbNmrnOb9OmDatXr1a/t0IIoQJLv/lX6JJsAD8/PyZPnvxQcYkQQlgySy/Jzjcw31+S3a9fP6ZPn05qaioHDhygWrVqNGnS5KH3nO4JeOEAACAASURBVD59mj59+jBy5Ej++ecfs3RcCCEKy9ILTPKdysgpyZ4xYwYtWrTgk08+YdmyZfz666+sX7/+ofObNWvGwYMHKVu2LIcPH2bs2LHs27fPLJ0XQpjG0ueCzcXqb/49qiR72bJlXLt2jb59+wIQFxeHp6cn27dv55lnnjG8t0uXLsyZM4fExMRcJdtCCMtQfG/+WXlgvr8ku379+hw7doznn3+eTZs2Gc5xcXEhICCAKlWqcPPmTapWrYpGoyEqKgq9Xi/rZQghLIplh2UTSrIfJywsjK1bt2Jra0upUqX48ssv5eagEMKiWPqIWUqyhSjGrHEqQ42S7Lfq+Rh97tro7SZfr6Ck8k8IUewoFj5iLnTlX6lSpZg1axYZGRnY2toye/ZsXnjhBe7evcvkyZO5fv06Op2O4cOH4+XlZe7PIYQQRrP6rAx4dOXfxIkTGTt2LF26dOHw4cN8/vnnfPvtt2zZsoUGDRrw9ddfk5iYSI8ePejduzf29vbm/ixCCGEUSy/JLnTln0ajISUlxXBOzv5/OccVRSElJYWKFStiZyczJkIIy6FXFKMfRaHQm7FOmzaNESNG8Omnn6LX6/n+++8BeP311xk9ejSdO3cmJSWFxYsXY2OTb/wXQognxrInMkzYjHXr1q1MnTqVw4cPM3XqVKZPnw7Azz//TNOmTTly5Ai7du1i7ty5JCcnm/2DCCGEsfQoRj+KQqE3Y925cyfdu3cHoGfPnkRFRQGwY8cOunfvjkajoW7dutSuXduwX6AQoniw9M1YlQL8rygUejPWatWqceLECQAiIyOpV68eADVq1ODYsWMA/Pfff1y+fNmwNrMQoniw9JLsLBSjH0XBqAKTv/76i+nTp+eq/Pvnn39YsGABWVlZlCxZklmzZuHk5ER8fDxTp07l5s2bKIrCW2+9ZVhT43GkwEQIYSw1Cky86/Yx+tyAK7vzfD0iIoL58+ej1+vx8fFh1KhRuV7fsGED27dvx9bWlipVqrBgwQJq1co75knlnxDFWHGt/PMsQGDekUdg1ul0uLm5sWHDBrRaLd7e3nz55Zc0bNjQcE5kZCQtWrSgdOnSfPfdd5w4cYKvvvoqz2tKuoQQothRFMXoR16ioqKoW7cujo6O2Nvb4+HhQXh4eK5z2rdvT+nSpQFo2bIlcXFx+fZPArMQothRKysjPj6e6tWrG55rtVri4+Mfe35AQADOzs759i/fPOZLly7x3nvvGZ7HxMQwYcIEbt++TXh4ODY2Njg4OODn54dWq+XixYtMmzaNc+fO8d577zFixIh8OyGEKBqWfpPOXApSkr1t2za2bdtmeO7r64uvr2+BrxkYGMjZs2fZvHlzvufmG5jr169PYGAgkD2f4uzsjKurKxUrVmTixIkA+Pv7s2LFCubOnUulSpWYPn36Q8N5IYTlscY5ZjUUJD85r0Cs1WpzTU3Ex8ej1WofOu+XX37h66+/ZvPmzUYtT1GgqYxjx47h6OhIrVq1KFeunOF4WlqaYc1lBwcHXnjhBSnDFkJYLLXmmJs3b050dDQxMTFkZmYSHByMi4tLrnP+/PNPZs6cyapVq3BwcDCqfwWKnsHBwfTq1cvwfPHixezatYvy5cvj7+9fkKaEEKLIqLWIkZ2dHTNnzmTkyJHodDq8vLxo1KgRS5YswcnJiW7duvHZZ5+RmprKu+++C2TXenz99dd5tmt0ulxmZiadO3cmODiYqlWr5npt9erVZGRkMGHCBMOxZcuWUaZMGaPmmCVdToiiYc4KPXNNZ6iRLtfdsYfR5+6LCTX5egVl9FRGREQEzZo1eygoA/Tu3Vt2whZCGFjDHLMlr5Vh9FRGcHAwHh4ehufR0dGGMuzw8HDq16+veueEEOZl6QHUXHSKZa/IbFRgTk1N5ZdffmHu3LmGY4sWLeLy5ctoNBpq1arFnDlzALh58yZeXl4kJydjY2PDpk2bCAkJyXWzUAhhGYprVoalby0lJdlCFGPWGJjVmGN2rtXN6HMj/n3yqb+S0yaEKHaKfDSaj0JX/p0+fZrLly8D2VtLlS9f3lCIAnD9+nU8PDwYN26cVP8JISxKUd3UM1ahK/+GDRtmOGfhwoUPzSEvXLiQzp0te55JCFE8WX1gvt/9lX85FEVh7969bNq0yXDswIED1KpVizJlyqjXUyGEUImlZ2UUqCT7wco/gF9//RUHBwdD6lxKSgpr165l3LhxqnVSCCHUZPVbS+XIzMzk4MGD9OiRu2Jmz549uYL18uXLeeONNyhbtqx6vRRCmIW5sicsfs8/ldbKMBejpzIeVfmXlZXF/v372bFjh+HYmTNnCAsL44svvuDOnTvY2NhQsmRJBg8erG7PhRCqsPScY3N4auaYH6z8g+yl7OrXr59roejvvvvO8Oec9TIkKAthmawxj1kNFlC+kSejpjJyKv+6d++e63hISMhDwVoIISydDr3Rj6IglX9CFGPWOGJWo/LPSdve6HPPxkeafL2Ckso/IUSxY+lrZUhgFkIUO/qinyjIk1GBeePGjWzfvh2NRkPjxo3x8/Nj+/btbNq0iatXr3Ls2DGqVKkCwLp16wgKCgKyKwUvXrzIsWPHqFSpkvk+hRBCFIDVj5jj4+Px9/cnJCSEUqVK8e677xIcHEzr1q3p2rUrQ4cOzXX+yJEjGTlyJAAHDx5k48aNEpSFEBblqRgx63Q60tPTsbOzIz09nWrVqvH888/n+75HVQoKIURRs/qSbK1Wy/Dhw3nllVfo1KkT5cqVo1OnTvk2nJaWxpEjRx5KsRNCWA5Lzzc2F0svyc53xJyUlER4eDjh4eGUL1+ed999l8DAQPr27Zvn+w4dOkTr1q1lGkMIC1ccg7Ni4SPmfAPzL7/8Qu3atQ0397p3786pU6fyDcyPqhQUQlgWa8xjVoOll2TnO5VRs2ZNzpw5Q1paGoqicOzYMRo0aJDne+7evcvJkyfp1s347VuEEOJJsfRFjPINzC1atMDNzY3+/fvTu3dv9Ho9vr6++Pv74+zsTFxcHH369GH69OmG9+zfv5+OHTvKesxCCIukRzH6URSkJFuIYswapzLUKMmuUSn/rLIcsbf/NPl6BSWVf0KIYsfSC0yMWl1u48aNeHh40KtXL95//30yMjJQFIXFixfj5uZGz5498ff3B7Lnbj755BNcXV3p3bs3586dM+sHEEKIgrL0OeZCV/4pikJsbCx79+7FxsaGhIQEIHtB/ejoaPbt28eZM2eYPXs227dvN/sHEUIIY1l9Vgb8X+VfVlaWofJv69atjB07Fhub7CYcHBwACA8Pp1+/fmg0Glq2bMmdO3e4ceOG+T6BEEIUkKWPmAtd+RcTE0NISAienp6MHDmS6OhoIHuEff+OJtWrVyc+Pt5sH0AIIQpKp9cb/SgK+Qbm+yv/jhw5QlpaGoGBgWRmZlKyZEl27NjBa6+9xrRp055Ef4UQVsDSN2O19HS5fAPz/ZV/JUqUMFT+abVaXF1dAXB1deV///sfkD3CjouLM7w/Li4OrVZrpu4LISyRpVf+Wf1UxuMq/1599VWOHz8OwIkTJ6hXrx4ALi4u7Nq1C0VROH36NOXLl6datWpm/RBCCFEQekUx+lEU8s3KuL/yz87OjqZNm+Lr60t6ejqTJk1i06ZNlClThvnz5wPQpUsXDh8+jKurK6VLl2bBggVm/xBCCFEQlp7HLJV/QhRjxbXyr3Tpukafm5Z2xeTrFZRR6XJCCPE00St6ox/5iYiIwM3NDVdXV9asWfPQ65mZmUycOBFXV1d8fHy4du1avm1KYBZCFDtq3fzT6XTMnTuXdevWERwczJ49e7hw4UKuc7Zv306FChXYv38/w4YN44svvsi3f7JWhhDFmKVnT5iLWjO4UVFR1K1bF0dHRwA8PDwIDw+nYcOGhnMOHjzIuHHjAHBzc2Pu3LkoioJGo3lsuxYRmNWYMxJCCGPdK0DM2bZtG9u2bTM89/X1xdfXF3i4oE6r1RIVFZXr/fHx8dSoUQMAOzs7ypcvz61btwybjzyKRQRmIYSwVPcH4idF5piFEKKQHiyoi4+Pf6igTqvVEhsbC0BWVhZ3796lcuXKebYrgVkIIQqpefPmREdHExMTQ2ZmJsHBwbi4uOQ6x8XFhZ07dwIQFhZG+/bt85xfBgvJYxZCCGt1+PBhFixYgE6nw8vLi9GjR7NkyRKcnJzo1q0bGRkZTJ48mb/++ouKFSuyePFiw83Cx5HALIQQFkamMoQQwsJIYBZCCAsjgVmIJ+S3334z6pgQEpiFVbt58ybh4eEcPHiQmzdvFnV38vTJJ58YdUwIiy4wmTdvXp5pJR9//HGh2m3VqlWe7f7++++Favd+V69epXr16tjb23P8+HH+97//0a9fPypUqGBSu//99x9ffvklN27cYN26dVy4cIFTp07h4+NjUrvbt2/P1YZOp2PVqlWGUlJL6y9k93nFihW0b9/esDv7mDFj8Pb2NrntzMxMwsLC+Pfff8nKyjIcL8zP49SpU5w6dYrExEQ2bNhgOJ6cnIxOpzO5rzl+/fVXrly5gpeXF4mJiaSkpOR79z8/iYmJ/PDDDw/9HPz8/Ard5r59+/J8vXv37oVu+2lh0YHZyckJyA6UFy5cwN3dHYDQ0FAaNGhQ6HZPnToFwFdffcUzzzxD3759Adi9e7dqo67x48fz448/cuXKFWbOnImLiwsffPABa9euNandjz76CE9PT77++msA6tWrx3vvvWdyoIuMjGTfvn3Mnz+fpKQkPvroI1566SWT2jRnfwHWrVvHzp07Dcn6t27dYsCAAaoE5tGjR1O+fHmaNWuGvb29SW3du3eP1NRUdDodKSkphuPlypVj6dKlpnYVgOXLl3P27FkuX76Ml5cX9+7dY/LkyXz//fcmtTtmzBhefPFFOnTogK2trSp9PXToEAAJCQmcOnWK9u3bA3D8+HFatWolgRlAsQI+Pj7KvXv3DM8zMzMVHx8fk9vt3bu3UccKo1+/foqiKMratWsVf39/RVEUpW/fvia36+np+VBbffr0MbldRVGU4OBg5aWXXlK6du2q/Prrr6q0ac7++vr6KhkZGYbnGRkZiq+vrypte3h4qNLO/a5du6YoiqKkpqaq3nafPn0UvV6f6+fcq1cvVdo1lzfffFOJj483PI+Pj1eGDx9ututZE6uYY05KSiI5OdnwPDU1laSkJJPbLVOmDLt370an06HX69m9ezdlypQxuV3IXqxkz5497Nq1i65duwLk+lWwsMqUKcOtW7cMUzE523eZKjo6Gn9/f9zc3KhZsyaBgYGkpaWZ3K65+gtQp04dXnvtNZYtW8by5cvx9fWlXr16bNiwIdeUQWG0atXKsI+lWm7cuIG7uzs9e/YE4Pz588yePVuVtkuUKIFGozH8nFNTU1Vpt2vXrhw+fFiVth4UGxuba9u5qlWrcv36dbNcy9pYRYHJjz/+yPLly2nXrh2KonDy5EnGjx9P//79TWr32rVrzJ8/n99//x2NRkPr1q2ZNm0atWvXNrnPFy5c4Pvvv6dly5b06tWLmJgY9u7dy6hRo0xq99y5c8ybN49//vmHRo0acevWLZYsWUKTJk1MardHjx7MnDmTl19+GUVR2LBhAz/++CPBwcEW2V/I/vU9L4WZD+7duzeQPcd+5coVateunWsqIygoqMBt5vDx8WHp0qWMHj2aXbt2AdCrVy/27NlT6DZzfPPNN1y5coWjR4/y9ttv8+OPP9KrVy+GDBliUrutWrUiLS0Ne3t77OzsDMtVqnEfZu7cuVy5cgUPDw8AQkJCqFu3LjNmzDC5bWtnFYEZsu++nzlzBsjeh/CZZ54p4h4ZLykpidjYWFWCEWSPvC9fvoyiKDz77LOUKFHC5DaTk5MpV65crmOXL1/m2WefNbltc/T3QUlJSVSoUCHfNQjy8++/eS8HWatW4bdB8/HxYfv27fTr188QmPv06cPu3bsL3SZkry0cFxfHpUuX+PnnnwHo1KkTHTt2NKndJ2H//v2cPHkSgLZt2+Lq6lrEPbIMFn3z79y5c7me56xpeuPGDW7cuEGzZs1Mav/y5cvMnj2bhIQE9uzZw/nz5zl48CBjxowxqV2AIUOGsGrVKrKysvD09MTBwYHWrVszdepUk9rV6XQcPnyYf//9F51Ox9GjRwF48803TWo3PT2dBQsWEB8fzzfffGPInjA1MD94Bz46Opry5cvTuHFjHBwcCtXm8uXL6dmzJw0aNCAzM5ORI0dy/vx5bG1tWbRoES+//HKh+5sTeE+fPk3Dhg0N/1glJydz8eJFkwJzjRo1DL+d3bt3D39/f5NuYufQaDSMGjWKoKAg1YLxxYsXadCgwUPfwRymfvdyPP/885QtW5aXX36ZtLS0Rw4QiiOLDswLFy587GsajQZ/f3+T2p8xYwZTpkxh5syZADRp0oRJkyapEpjv3r1LuXLlDCOkCRMmGH5NNsU777xDyZIlady4MTY26t0iMFf2REBAAKdPn6Zdu3YAnDhxgmbNmnHt2jXGjBlDv379Ctzm3r17GTt2LAA7d+5EURSOHTtGdHQ0H374oUmBOcfs2bMNK4JB9lz5g8cK0+b8+fOJj4/H2dmZjh07MmvWLJP7CtkBLioqihdeeEGV9jZu3Mi8efMe+R1U47sH8MMPP7Bt2zaSkpI4cOAA8fHxzJo1i02bNpnctrWz6MD87bffotfrOXXqFC+++KLq7aelpT30F1mtlCCdTseNGzfYu3cvEydOVKVNgLi4OJPmOR/n1q1buLu7GzaTtLOzUyXw63Q6QkJCqFq1KpCd1/zhhx/yww8/MHjw4EIF5pwbXQA///wzHh4e2Nra0qBBA9XygpUHtv6xsbEx+ebtH3/8waJFi3Id27p1KwMHDjSpXYAzZ84QFBREzZo1KV26tOF4Yf+uzJs3D8j+DprLli1b2L59O6+99hqQPRhITEw02/WsiUUHZsj+QsybN88wJ6emypUrc/XqVcMXMDQ0VLW56zFjxjBixAhefPFFXnjhBWJiYqhXr57J7To7O/Pzzz/TqVMn0zt5H3NlT8TGxhqCMoCDgwOxsbFUqlQJO7vC/fWzt7fn77//pmrVqhw/fpwpU6YYXlMjkwTA0dERf39/Q9D87rvvTC7WWLVqFfb29nTo0AHIzsOOjIxUJTB/8803JrfxKBkZGXz33Xf89ttvaDQaXnzxRQYOHEjJkiVNbtve3j7XjVU1spaeFhYfmAE6dOhAWFgY3bt3N/nmzv1mzZrFjBkzuHTpEp07d6Z27dpG7WBrjJ49exrSoiD7i75s2TKT223ZsiXjxo1Dr9erepf8o48+YvTo0Vy9epUBAwYYsidM9dJLL/H222/To0cPIHuh8JdeeonU1NRCB/7p06czYcIEbt26xRtvvGEImIcPH+b55583uc8Ac+bM4ZNPPmHVqlVoNBo6dOhgGEUW1sqVK3nnnXcoUaIER44c4dKlS6xcuVKV/ubMfSckJJCRkaFKmwBTpkyhbNmyDB48GIA9e/YwefJkVQpj2rZty9dff016ejpHjx7lu+++e2iR+eLKKrIyclJ2bG1tKVmypGrBSKfTYWtrS2pqKnq9XtWbDhkZGQQEBPDPP//k+qKYUsoK2bshrFy5kueee06Vf6SioqKoUaMGzzzzDFlZWWzbto2wsDAaNmzIhAkTqFSpkkntK4rCvn37DIv1VKhQgYSEBNXmVs1Bp9MxZcqUh6Yd1JCQkMCwYcNwcnJiwYIFqg00wsPD+fTTT7lx4wZVqlTh+vXrNGjQwOR0R3d3d0JCQvI9Vhh6vZ6AgIBcmSQ50xrFnVWMmHNKqNXWrVs3OnfujLu7u6EsVC2TJ0+mfv36/Pzzz4wdO5agoCDq169vcrs1atSgcePGqn2hZ82aZSjGOHXqFKtWrWLGjBn89ddfzJw50+SRkUajwdHRkdOnTxMWFkatWrVwc3NTo+vcunWLFStWGH7Nbt26NWPHjs13P7X82Nracv36dTIzM00ux4b/W5slZ0Bx7949rl27RmhoqGo5wUuWLGHbtm28+eab7Nq1i8jISJPT8CD7puLp06dp2bIlkD2XnbNUgqmWLVvGu+++awjGOp2ODz74wCz/IFobqwjMiqKwe/durl27xtixY4mNjeXmzZsm34Heu3cvhw4dYsuWLUyfPp2uXbvi7u5OmzZtTO7z1atXWbp0KeHh4fTv359evXrx+uuvm9yuo6MjQ4YMwdnZOVfQKGy6nE6nM4yKQ0JC8PX1xc3NDTc3N8MaIoVx+fJlgoOD2bNnD5UrV8bd3R1FUVS9mfT+++/Tpk0bwz8eQUFBvPfee2zcuNHkth0dHRk4cCAuLi65qkEL83M218DifnZ2dlSuXBm9Xo9er6d9+/YsWLCg0O3lZBBlZWUxYMAAatasCcD169dVGWBA9o3s1atX8/bbb5OZmcnEiRNp2rSpKm1bO6sIzLNnz8bGxobIyEjGjh1LmTJlmDNnDj/++KNJ7ZYuXRp3d3fc3d1JSkpi/vz5DBkyhL/++svkPufc2KpQoYLhRlVCQoLJ7dauXZvatWtz79497t27Z3J7er2erKws7OzsOHbsWK55VFMyHHr27EmbNm1YvXo1devWBVAlYN7v5s2bhrQ5yL7hunfvXlXarlOnDnXq1EFRlFwLD5li//79tG/f3jC3fufOHU6cOMGrr75qctsVKlQgJSWFtm3bMmnSJKpUqWLS8gI5aZPmtGDBAiZNmsTq1as5fvw4zs7ODBs2zOzXtQZWEZijoqLYuXOnIbWqYsWKqgQlyM6rDQkJ4ciRIzg5OfHVV1+p0q6vry9JSUm8++67jB49mtTUVCZMmGByu6Yuw/kgDw8PBg8eTOXKlSlVqpTht4UrV66YNOe+fPlygoODGTp0KJ07d8bDwwO1b2d07NiR4OBgw03W0NBQ1bJV1P45Q/bP5P7KtgoVKrB8+XKTAvP169epWbMmK1eupFSpUkydOpWgoCDu3r2b6x+tgnqwkEbNm4r3F60MHTqUmTNn0rp1a9q2bcu5c+dUK16xZlZx88/Hx4fvv/8eb29vdu7cSWJiIsOHDzc5hc7FxYWmTZvSs2fPh35ltVSJiYmsXbuWCxcu5PqimJLwf/r0aW7evEnHjh0NP4PLly+Tmppq8pckNTWV8PBwgoODiYyMpG/fvri6upoUQO+fs825KQzZI/wyZcqoMmdrjp9z7969H8orftSxgujfv7+h6GX8+PGqZP7czxw3FfNav0Ot4hVrZxUj5iFDhjB27FgSEhJYvHgxoaGhqhRt7N69W/Xyz/xWNTO1dHrSpEn07NmTn376iTlz5rBz506qVKliUps5N3bup8YaGZCdH927d2969+5NUlISoaGhrF271qTA/CTmbM3xc3ZycsLPz89wr2HLli0m/8N3/7gqJibGpLYexRw3FXMKx0JDQw1rrIvcrCIw9+nTh2bNmhEZGYmiKKxcudKkNQbWrl3LW2+9xeLFix+Z3VDYnVEA1eYjH+f27dv4+Pjg7+/PSy+9xEsvvYSXl5dZr6mWihUr4uvri6+vr0ntPIl1HMzxc54xYwYrV640DCo6duxoWA6gsO7/+6tmjn8OtW8q5rCxsWHdunUSmB/DKgLz7du3cXBwMCwPCNm7QhR2lbKcoK5W2s/9zDE3eb+cm4rVqlXjp59+olq1aqqsTW1NHrWOw/1BSY1fhc3xcy5TpgyTJk0yuW/3O3/+PK1bt0ZRFDIyMmjdujWAarn+at9UvN/LL7/MN998g7u7e64yclNz558GVjHH7OLiQmxsrGG/vDt37lC1alWqVq3KvHnzCh1gzXmj4cMPP2T69OmGPiclJbFw4UKTC0wOHTpEmzZtiI2NZd68eaSkpDB27Fi6deumRretwv1FMZC9kFFYWBi1a9dm3Lhxqnyx1fw5z507l5kzZ/LOO+888vUnkQFRUFeuXOG///6jadOmlCpVCr1eT1BQEP/++y9du3ZVZVDzqCo/jUZDeHi4yW1bvSewS4rJpk+frkRERBieHzlyRJkxY4Zy6tQpxdvbu9DtDh48WOnRo4eyePFi5X//+58aXTV41DZSamwtJbK37bp165aiKIpy4sQJpWPHjkpoaKiyePFiZfz48Sa1nZ6ermzYsEGZM2eOsnXr1lxbmhVWq1atFEVRlOPHjz/yYYlGjRqlnD9//qHj58+fV95+++0i6FHxYhVTGWfOnMm1zXunTp349NNPmTt3LpmZmYVu99tvv+XmzZvs3buXmTNnkpKSQs+ePVVZ9lOv15OUlETFihWB7OkYU/KC89qtQ6PRmJQaZW3MVRQD2b/p2NnZ0aZNGyIiIrhw4YJJ9xwgOycaUGVz2yflv//+47nnnnvo+HPPPZfvZgIF8ffff3PhwoVc3+PCrDj4tLGKwPzMM8+wZs2aXFvQVK1aFZ1OZ/LSlM888wxDhw6lXbt2rFu3jpUrV6oSmIcPH85rr72WK8f2cb/KGuNR83qpqan8+OOP3L59u1gFZnMVxUD2jcWc9DVvb29VdvNOTEzMM1vH1Ewdc7h79+5jX0tPT1flGsuXL+f48eNcvHiRLl26EBERwYsvviiBGSsJzF988QUrVqwwBJ/WrVuzaNEidDqdSQUhFy9eJCQkhH379lGpUiV69uzJRx99pEqf+/Xrh5OTE5GRkUD2X8KGDRsWur3hw4cb/pycnIy/vz87duzA3d0912vFgbmKYoBcS5EWdlnSB+n1erNn66jNycmJH3744aFFhbZv367afZmwsDACAwPp168ffn5+/Pfff0yePFmVtq2dVdz8MxdfX1/c3d3p0aMHWq1WlTYzMjLYunUrV69epXHjxnh7e6v2Bb99+zYbNmwgKCiI/v37M3ToUMNUSXFjrqKYpk2bGjIElP+f6VCqVCmTshzuLwKxFv/99x/jxo2jRIkShp/n2bNnbD5lXgAABghJREFUuXfvHsuXL1dl3XJvb28CAgLw9PTE39+fsmXL0rNnT0JDQ01u29pZxYj58uXLrF+/nn///TfXYtqmpEXpdDpq167NG2+8oUYXDR6co7x48SLTp083ud1PP/2U/fv389prrxEUFETZsmVV6K31MldRjBrrpDzIGsc+VatW5fvvvycyMpJ//vkHgC5duhgW+VeDk5MTd+7cwcfHB09PT8qUKUOrVq1Ua9+aWcWIuU+fPgwYMAAnJ6dcc8qmpuwMGjSIjRs3qrK0Y477S2yzsrLw8fFRZbTUpEkT7O3tsbW1zZWza8pITjwZt2/fltzcfFy7do3k5GTVdpK3dlYxYrazs2PQoEGqt1u7dm3VlnbMYY45SsguJBDWSYLy4+VsopCzbZUE5mxWMWJetmwZVapUwdXVNdfo1tS/8I9LQTOles8cc5RCPI1mz57N1atXc2Vb1alTx6J3t3lSrCIwS4WQEE+fHj16sHfvXsPUnF6vx8PDQ7U1ta2ZVUxlHDx40CztDhky5JELv8iyg0KYX926dbl+/bph7efY2FjDpgrFnVUE5rS0NDZs2GBYtyA6OprLly/zyiuvmNTuhx9+aPhzRkYG+/btM6ztK4Qwj5xCq5SUFNzd3Q1bxEVFRZm8XdzTwioC89SpU2nWrJlhHV6tVsu7775rcmB+MKvjxRdfxNvb26Q2hRB5K24FUYVhFYH56tWrfPXVV4ZdE0qXLq1Kbujt27cNf9br9Zw9ezbPUlQhhOkeXDMkOTk5V32CsJLAbG9vT3p6umE++OrVq6rkHnt6ehratLOzo1atWsyfP9/kdoUQ+du2bRtLly6lZMmShq3C5KZ+NqvIyjh69CirVq3iwoULdOzYkVOnTuHn50e7du0K1d6TWM9XCJG37t278/3335u8ZdfTyCoCM8CtW7c4c+YMiqLQokULk/5j9u/fnw0bNlCpUiVOnjzJe++9x4wZM/jrr7+4dOkSS5cuVbHnQohHGTFiBMuXL8+1e4nIZhVTGb/99htNmzala9euBAYGsnr1aoYOHfrQFuvGMud6vkII43zwwQcMGDCAFi1a5JqaNHX966eBaYsZPyGzZ8+mdOnSnD9/no0bN1KnTp1cqW4FlbOeL8CxY8do37694TVT1/MVQhhn5syZtG/fnhYtWtCsWTPDQ1jJiNnOzg6NRsOBAwcYNGgQPj4+BAQEFLo9c67nK4QwTlZWFlOnTi3qblgkqwjMZcuWZfXq1QQFBbF58+ZcI97CGD16NB06dDCs53t/SeiMGTPU6rYQIg/Ozs5s27aNV155RdU1cJ4GVnHz7+bNm+zZs4fmzZvTpk0brl+/zokTJ2QLGiGsmKyB83hWEZhTU1MpWbIktra2XL58mUuXLuHs7EyJEiWKumtCCKE6q7j5N3jwYDIzM4mPj2fEiBEEBgaqtjefEOLJWrt2reHPD64k9+WXXz7p7lgkqwjMiqJQunRp9u3bx8CBA1m6dKlhuxshhHUJCQkx/HnNmjW5Xjty5MiT7o5FsprAfOrUKYKCgujatavhmBDC+tz/3X3weyzf62xWEZinT5/O6tWrefXVV2nUqBExMTGFLscWQhSt+9dAf3A99Eetj14cWcXNPyHE0yNn+7X7t16D7NFyZmYm586dK+IeFj2rCMyJiYmsXbuWCxcukJGRYTguO40IIZ5GVjGVMWnSJOrXr8+1a9cYN24ctWrVonnz5kXdLSGEMAurCMy3b9/Gx8cHOzs7XnrpJfz8/IiMjCzqbgkhhFlYRUm2nV12N6tVq8ZPP/1EtWrVSEpKKuJeCSGEeVjFHPOhQ4do06aNYTPWlJQUxo4dS7du3Yq6a0IIoTqLDswZGRls3bqVq1ev0rhxY7y9vQ2jZyGEeFpZdGCeOHEidnZ2tGnThoiICGrWrCmLaAshnnoWPfy8ePEiQUFBAHh7e+Pj41PEPRJCCPOz6KyM+6ctZApDCFFcWPRURk6FEJCrSihnm/Pff/+9iHsohBDqs+jALIQQxZFFT2UIIURxJIFZCCEsjARmIYSwMBKYhRDCwkhgFkIIC/P/AAUNNJdVjUrgAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.heatmap(train_df.isnull())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 339
        },
        "id": "qke46m4MOTx0",
        "outputId": "0504994d-8df1-4cd4-ff60-1a3eb3ddd320"
      },
      "id": "qke46m4MOTx0",
      "execution_count": 247,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fe1f7f13c50>"
            ]
          },
          "metadata": {},
          "execution_count": 247
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXsAAAEwCAYAAABWodGkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzde1xU1fr48c8A4g0BpRi8oCZpXistLxzIFA+SICkoUR3pYJp5yzBFU/OSN8wsNTWPZKaWKWriDRQVvOetk4pa1jE1sWAoUBBBgZn5/cGP/XVEZWA2yuV5n9d+HfeeNWuvkVysWXs9z9IYjUYjQgghKjWrR90AIYQQZU86eyGEqAKksxdCiCpAOnshhKgCpLMXQogqQDp7IYSoAsqssz9w4AA+Pj54e3sTGRlZVrcRQghhhjLp7PV6PdOnT2f58uXExMSwfft2Lly4UBa3EkKIR6q4ge2JEycICAigdevW7Ny50+S16OhoevbsSc+ePYmOjlaunz17Fn9/f7y9vZk5cyZqhEOVSWefmJhIkyZNcHV1xdbWFj8/P+Lj48viVkII8ciYM7CtX78+ERER9O7d2+T69evXWbx4MevXr2fDhg0sXryYjIwMAKZNm8aMGTPYtWsXly9f5sCBAxa3tUw6e51Oh4uLi3Ku1WrR6XRlcSshhHhkzBnYNmrUiJYtW2JlZdrdHjp0CA8PDxwdHXFwcMDDw4ODBw+SmppKVlYWzz77LBqNhr59+6oyWJYHtEIIUUqWDGzv9967r7u4uKgyWLaxuIZ70Gq1pKSkKOc6nQ6tVnvf8j806lsWzRBCVELPX91scR15f180u+ym+BNERUUp58HBwQQHB1vchoetTDr7du3acfnyZZKSktBqtcTExPDJJ5+Uxa2EEKLkDHqziz6ocy/pwPbu9x4/ftzkvZ06dSpSZ0pKitl1PkiZTOPY2NgwZcoUBg8ejK+vL7169aJ58+ZlcSshhCg5o8H84wHuHNjm5uYSExODl5eXWU3w9PTk0KFDZGRkkJGRwaFDh/D09MTZ2Rk7OztOnTqF0Whk8+bN9OjRw+KPrCkPKY5lGkcIYS5VpnGSfza7bLX6rR74+v79+5k9ezZ6vZ5+/foxbNgwFi5cSNu2benRoweJiYmMHDmSzMxMqlevzmOPPUZMTAwAGzduZNmyZQAMHTqUfv36AXDmzBkmTJjArVu36Nq1K5MnT0aj0ZTy0xaQzl4IUaGo0dnn/nnO7LK2DdpYfL/yoEzm7IUQolzT5z/qFjx0Fnf2hV9dtFoty5YtY+LEiZw9exaj0cgTTzxBREQEtWvXVqOtQgihjhI8oK0sLH5Au3r1atzc3JTziRMnsnXrVrZt20b9+vVZs2aNpbcQQgh1qfSAtiKxqLNPSUlh37599O/fX7lmZ2cHgNFo5NatW5a1TgghyoLBYP5RSVjU2c+ePZvw8PAiYcATJkzAw8ODixcvEhISYlEDhRBCbUajweyjsih1Z793717q1atH27Zti7wWERHBwYMHcXNzIzY21qIGCiGE6mRkb74ff/yRhIQEvLy8eO+99zh69Chjx45VXre2tsbPz49du3ap0lAhhFCNPs/8o5Io9WqcMWPGMGbMGACOHTvGihUr+Pjjj/n9999p0qQJRqORhIQEmjVrplpjhRBCFZVoesZcqq6zNxqNjB8/nps3b2I0Gnnqqaf48MMP1byFEEJYrhJNz5hLImiFEBWKGhG0t8/uNrts9bbeFt+vPJAIWiFE1VMFR/YWdfYrV65kw4YNaDQaWrRoQUREBLa2tixYsICdO3diZWXFa6+9xhtvvKFWe4UQwmJGQ+V58GquUnf2Op2O1atXExsbS40aNXj33XeJiYnBaDSSnJzMjh07sLKyIi0tTc32CiGE5argyN6ioCq9Xs+tW7fIz8/n1q1bODs7s3btWkaMGKEEWjk5OanSUCGEUI2kSzCfVqvlzTffpHv37nh6emJnZ4enpydJSUnExsYSGBjI4MGDuXz5sorNFUIIFRj05h+VRKk7+4yMDOLj44mPj+fgwYPk5OSwZcsWcnNzqV69Ops2beKVV15h4sSJarZXCCEsJyN7833//fc0atSIevXqUa1aNXr27MnJkyfRarV4excsVfL29uaXX35RrbFCCKGKKpguodQPaBs0aMDp06fJycmhRo0aHDlyhLZt22JnZ8exY8dwdXXl+PHjNG3aVMXmCiGECmTzEvM988wz+Pj4EBAQgI2NDa1atSI4OJhbt24xduxYVq1aRa1atZg1a5aa7RVCCMupOGI/cOAAs2bNwmAwEBQUxJAhQ0xez83NZdy4cZw7dw5HR0fmz59Po0aN2Lp1K19++aVS7pdffiE6OppWrVoREhJCamoqNWrUAGDFihUWL3aRCFohRIWiRgRtzoGVZpet2TX0vq/p9Xp8fHz46quv0Gq19O/fn08//ZQnn3xSKbNmzRp++eUXpk+fTkxMDLt372bBggUm9fzyyy+MGDGCPXv2ABASEsK4ceNo165diT7Xg1i8U5UQQlQ4Ks3ZJyYm0qRJE1xdXbG1tcXPz4/4+HiTMgkJCQQEBADg4+PDkSNHuHuMHRMTg5+fn7qf8S7S2Qshqh6VVuPodDpcXFyUc61Wi06nK1Kmfv36ANjY2FCnTh2uXbtmUiY2NrZIZz9x4kT69OnDkiVLivxyKI1i5+wnTJjAvn37cHJyYvv27QB89NFH7N27l2rVqtG4cWMiIiKwt7cnNzeXqVOncvbsWTQaDZMmTaJz584WN1IIIVRVgjn7qKgooqKilPPg4GCCg4NVa8rp06epWbMmLVq0UK7NmzcPrVZLVlYWo0aNYsuWLfTta9l0d7Ej+8DAQJYvX25yzcPDg+3bt7Nt2zaaNm3KsmXLANiwYQMA27Zt46uvvuKjjz7CUImWLgkhKgl9vtlHcHAwmzZtUo47O3qtVktKSopyrtPp0Gq1JrfSarUkJycDkJ+fz40bN6hbt67y+r2mcArrsLOzo3fv3iQmJlr8kYvt7Dt27IiDg4PJNU9PT2xsCr4UPPvss8qHvXDhgjKSd3Jyok6dOpw9e9biRgohhKpUmsZp164dly9fJikpidzcXGJiYvDy8jIp4+XlRXR0NABxcXF06dIFjUYDgMFgYMeOHSadfX5+Punp6QDk5eWxb98+mjdvbvFHtjjF8XfffUevXr0AaNmyJQkJCfTu3Zvk5GTOnTtHcnIyTz/9tMUNFUII1ag042BjY8OUKVMYPHgwer2efv360bx5cxYuXEjbtm3p0aMH/fv3Jzw8HG9vbxwcHJg/f77y/hMnTlC/fn1cXV2Va7m5uQwePJi8vDwMBgPu7u688sorlrfVkjcvXboUa2trXn75ZQD69evHb7/9Rr9+/WjQoAHt27fH2tra4kYKIYSqVJxefvHFF3nxxRdNrr377rvKn6tXr85nn312z/d27tyZ9evXm1yrVasWmzZtUq19hUrd2W/atIl9+/axcuVK5SuJjY2NSS6cV199VSJohRDlTyXKeWOuUnX2Bw4cYPny5XzzzTfUrFlTuZ6Tk4PRaKRWrVocPnwYa2trk+ACIYQoFyRdQlHvvfcex48f59q1a3Tt2pV33nmHyMhIcnNzGThwIFCQOmH69OmkpaUxaNAgrKys0Gq1zJ07t8w/gBBClFgVXCUo6RKEEBWKKukSNs02u2zNwMqRpl02HBdCVD1VcGRfqgjasLAwLl26BMCNGzeoU6cOW7Zs4fDhw3zyySfk5eVRrVo1wsPDcXd3L9tPIIQQJSWdfVGBgYEMGDCA8ePHK9fuzNg2Z84c7OzsAKhbty5Lly5Fq9Xy66+/MmjQIA4ePFgGzRZCCAs8+tnrh67Yzr5jx45cvXr1nq8ZjUZ27NjBqlWrAGjdurXyWvPmzbl9+za5ubnY2tqq1FwhhFBBvqzGKZEffvgBJyene66lj4uLo3Xr1tLRCyHKH1lnXzLbt2+nd+/eRa7/73//Y968eaxYscKS6oUQomxUwTn7Uuezz8/PZ/fu3fj6+ppcT0lJYeTIkXz00Uc0btzY4gYKIYTqjEbzj0qi1CP777//nmbNmpkk7s/MzGTIkCGMGTOG5557TpUGCiGE6mRkX9R7773Hq6++yqVLl+jatauSs/5eO6t88803XLlyhSVLltCnTx/69OlDWlpa2bRcCCFKS6VtCSsSiaAVQlQoakTQZkeONrtsrSHziy9UAUgErRCi6qlEI3ZzSWcvhKh6quDSy2Ln7JOTkwkJCcHX1xc/Pz8lgKpwK62WLVty5swZpfzVq1d5+umnlTn7KVOmlF3rhRCiNAxG849KotiRvbW1Ne+//z5t2rQhKyuLfv364eHhQYsWLVi0aBFTp04t8p7GjRuzZcuWMmmwEEJYrApO4xQ7snd2dqZNmzZAwU7nzZo1Q6fT4ebmRrNmzcq8gUIIoTq93vyjGAcOHMDHxwdvb28iIyOLvJ6bm0tYWBje3t4EBQUp6WceNAty9uxZ/P398fb2ZubMmaixjqZEQVVXr17l559/5plnnim2XN++fRkwYAA//PCDRQ0UQgjVqbT0Uq/XM336dJYvX05MTAzbt2/nwoULJmU2bNiAvb09u3fvJjQ0lHnz5imvFc6CbNmyhenTpyvXp02bxowZM9i1axeXL1/mwIEDFn9kszv7mzdvMmrUKCZOnKhkubwXZ2dn9u7dy+bNm3n//fcZM2YMWVlZFjdUCCFUo9KcfWJiIk2aNMHV1RVbW1v8/PyIj483KZOQkEBAQAAAPj4+HDly5IEj9dTUVLKysnj22WfRaDT07du3SJ2lYVZnn5eXx6hRo/D396dnz54PLGtra0vdunUBaNu2LY0bN1Zy3wshRLlgNJh/PIBOpzPJIqDVatHpdEXK1K9fHwAbGxvq1KnDtWvXgHvPgtxdp4uLS5E6S6PYB7RGo5FJkybRrFkzZc/ZB0lPT8fBwQFra2uSkpK4fPkyrq6uFjdUCCFUU4JVNlFRUURFRSnnwcHBBAcHW9yEwlmQunXrcvbsWUaMGEFMTIzF9d5PsZ39f//7X7Zs2UKLFi3o06cPUJBCITc3lxkzZpCens7bb79Nq1at+PLLLzlx4gSfffYZNjY2WFlZ8eGHH+Lo6FhmH0AIIUrKWILVOA/q3LVaLSkpKcq5TqdDq9UWKZOcnIyLiwv5+fncuHGDunXrotFolBTwd86C3F1nSkpKkTpLo9jO/vnnn+eXX36552ve3t5Frvn4+ODj42Nxw4QQosyYscrGHO3atePy5cskJSWh1WqJiYnhk08+MSnj5eVFdHQ07du3Jy4uji5duqDRaO47C+Lo6IidnR2nTp3imWeeYfPmzYSEhFjcVomgFUJUPSoFS9nY2DBlyhQGDx6MXq+nX79+NG/enIULF9K2bVt69OhB//79CQ8Px9vbGwcHB+bPL8i186BZkKlTpzJhwgRu3bpF165d6dq1q8VtLTYRWnJyMuPGjSMtLQ2NRsMrr7zCv//9bxYtWsT69eupV68eUDC18+KLL5KXl8cHH3zATz/9RH5+Pn379uXtt99+YCMkEZoQwlxqJEK7Oe01s8vWnrbW4vuVB6WOoAUIDQ1l0KBBJuV37txJbm4u27ZtIycnBz8/P/z8/GjUqFHZfAIhhCipSpQGwVzFdvbOzs44OzsDphG096PRaMjJySE/P59bt25RrVq1B67LF0KIh04SoT3Y3RG0a9aswd/fnwkTJpCRkQEUPKCtWbMmnp6edO/enTfffFNW4wghypcqmAit1BG0r732Grt372bLli04OzszZ84coCCizMrKioMHDxIfH8+KFStISkoqsw8ghBAlZczXm31UFqWOoH3sscewtrbGysqKoKAgJc3x9u3beeGFF6hWrRpOTk506NDBJAWyEEI8cjKyL+p+EbSpqanKn/fs2UPz5s0BqF+/PseOHQMgOzub06dPS3ZMIUT5olK6hIqk1BG027dv5/z58wA0bNhQydj2r3/9iwkTJuDn54fRaCQwMJCWLVuW4UcQQogSqkQjdnPJhuNCiApFjXX2N8L8zS5bZ8E2i+9XHkgErRCi6qlED17NJZ29EKLqqYLTOMU+oL19+zb9+/fn5Zdfxs/Pj88++wyApKQkgoKC8Pb2JiwsjNzcXKAg30NAQACtW7dm586dZdt6IYQoDVmNU5StrS2rVq1i69atbN68mYMHD3Lq1CnmzZtHaGgou3fvxt7eno0bNwIFq3EiIiLo3bt3mTdeCCFKw2g0mn1UFsV29hqNhtq1awOQn59Pfn4+Go2Go0ePKqmMAwIClG2zGjVqRMuWLbGyKlFwrhBCPDxVcGRv1py9Xq8nMDCQK1eu8Prrr+Pq6oq9vT02NgVvV2vbLCGEeCgqUSduLrM6e2tra7Zs2UJmZiYjRozg4sWLZd0uIYQoM8b8yhMsZa4Srcaxt7enc+fOnDp1iszMTPLz87GxsVFt2ywhhHgoql5fX/ycfXp6OpmZmQDcunWL77//Hjc3Nzp37kxcXBwA0dHReHl5lW1LhRBCJUaD0eyjsih2ZJ+amsr777+PXq/HaDTy0ksv0b17d5588klGjx7NggULaNWqFUFBQUBB1suRI0eSmZnJ3r17WbRoUZnumC6EECVWiTpxc0m6BCFEhaJGuoTrwd3NLusYtdfi+5UHEkErhKhy1JyeOXDgALNmzcJgMBAUFMSQIUNMXs/NzWXcuHGcO3cOR0dH5s+fT6NGjTh8+DCffPIJeXl5VKtWjfDwcNzd3QEICQkhNTWVGjVqALBixQqcnJwsamepI2gLzZw5k/bt2xd5X1xcHE899ZTkshdClDvGfKPZx4Po9XqmT5/O8uXLiYmJYfv27Vy4cMGkzIYNG7C3t2f37t2EhoYyb948AOrWrcvSpUvZtm0bc+bMYdy4cSbvmzdvHlu2bGHLli0Wd/RgQQQtwJkzZ5TtCO+UlZXF6tWrle0LhRCiXDGU4HiAxMREmjRpgqurK7a2tvj5+SkBpoUSEhIICAgACrZtPXLkCEajkdatWyurGJs3b87t27eVtDNlodQRtHq9nrlz5xIeHl7kPQsXLuStt96ievXq6rdYCCEsVJK9S6KioggMDFSOqKgopR6dToeLi4tyrtVqiwSY6nQ66tevD4CNjQ116tTh2rVrJmXi4uJo3bo1tra2yrWJEyfSp08flixZokrahlJF0D7zzDOsWrWKHj164OzsbFL23LlzpKSk0K1bN7788kuLGyiEEKorwTr74OBggoODy6wp//vf/5g3bx4rVqxQrs2bNw+tVktWVhajRo1iy5Yt9O1r2UIWsxLYFEbQ7t+/n8TERE6cOMHOnTsZMGCASTmDwcCcOXMYP368RY0SQoiypNauhFqtlpSUFOVcp9MVCTDVarUkJycDBbMjN27coG7dugCkpKQwcuRIPvroIxo3bmzyHgA7Ozt69+5NYmKixZ+5RNnKCiNojx07xpUrV+jZsydeXl7k5OTg7e3NzZs3+fXXX3njjTfw8vLi1KlTDBs2TB7SCiHKFWO++ceDtGvXjsuXL5OUlERubi4xMTFFAky9vLyIjo4GCqZrunTpgkajITMzkyFDhjBmzBiee+45pXx+fj7p6ekA5OXlsW/fPmWPb0sUO42Tnp6OjY0N9vb2SgTtW2+9xeHDh5Uy7du3Z/fu3QDKZuNQsHxo3LhxtGvXzuKGCiGEWtTaR9zGxoYpU6YwePBg9Ho9/fr1o3nz5ixcuJC2bdvSo0cP+vfvT3h4ON7e3jg4ODB//nwAvvnmG65cucKSJUtYsmQJULDEsmbNmgwePJi8vDwMBgPu7u688sorFre12KCq8+fPF4mgHTlypEmZ9u3bc/LkySLvNbezl6AqIYS51Aiq0nV/0eyy2r37Lb5feSARtEKICkWVzr5bN7PLavfts/h+5YFE0Aohqhy1pnEqEunshRBVjtGgedRNeOiK7exv377Nv/71L3Jzc9Hr9fj4+DBq1CiOHDnC3LlzMRgM1KpVizlz5tCkSRNmz56tPKS9desWaWlp/PDDD2X+QYQQwlwGvXT2RRSmS6hduzZ5eXm8/vrrdO3alWnTpvH555/j5ubGmjVrWLp0KXPmzGHixInKe7/++mt++umnMv0AQghRUlVxGqfU6RKgIAdO4f/fHUkLEBMTQ+/evdVsrxBCWMxo0Jh9VBalTpcwa9YshgwZQvXq1bGzs2P9+vUm7/njjz+4evUqXbp0KZOGCyFEaT36NYgPX6nSJfz666+sXLmSyMhIDhw4QGBgIBERESbviYmJwcfHB2tr6zJpuBBClFZVHNmXKl3CgQMHOH/+vJLC2NfXt0hQVWxsLH5+fuq1VAghVGLQa8w+KotSbzh+48YNLl26BMDhw4dxc3NT3vPbb7+RmZl5z01NhBDiUauKI/tSbzg+c+ZMRo0ahUajwcHBgdmzZyvviY2NxdfXV3mQK4QQ5YnRWPX6JkmXIISoUNRIl3ChtY/ZZZ/8Kc7i+5UHEkErhKhyDFVwZG92Z1+YvlOr1bJs2TImTpzI2bNnMRqNPPHEE0RERFC7du377qQuhBDlRVWcxjF7Nc7q1atNHsJOnDiRrVu3sm3bNurXr8+aNWuA+++kLoQQ5YWsxrmPlJQU9u3bR//+/ZVrdnZ2ABiNRm7duqVcv99O6kIIUV5UxdU4ZnX2s2fPJjw8HCsr0+ITJkzAw8ODixcvEhISApi3k7oQQjxKBqPG7KOyKLaz37t3L/Xq1aNt27ZFXouIiODgwYO4ubkRGxtbJg0UQgi1GY0as4/KotjO/scffyQhIQEvLy/ee+89jh49ytixY5XXra2t8fPzY9euXcCDd1IXQojywGg0/yjOgQMH8PHxwdvbm8jIyCKv5+bmEhYWhre3N0FBQVy9elV5bdmyZXh7e+Pj48PBgwfNrrM0iu3sx4wZw4EDB0hISODTTz+lS5cufPzxx/z+++9AwZx9QkICzZo1A+6/k7oQQpQXak3j6PV6pk+fzvLly4mJiWH79u1cuHDBpMz9Fq1cuHCBmJgYYmJiWL58OR9++CF6vd6sOkujRLlxChmNRsaPH4+/vz/+/v6kpqYyYsQIAPr378/169fx9vbmq6++MvkWIIQQ5YHBoDH7eJDExESaNGmCq6srtra2+Pn5ER8fb1LmfotW4uPj8fPzw9bWFldXV5o0aUJiYqJZdZZGiYKqOnfuTOfOnQFYt27dPctUr16dzz77zOKGCSFEWVHrwatOp8PFxUU512q1JCYmFilzr0UrOp1OSSZZ+F6dTgdQbJ2lIRG0QogqpyQPXqOiooiKilLOg4ODCQ4OLotmlSnp7IUQVU5JRvYP6ty1Wi0pKSnKuU6nQ6vVFimTnJyMi4uLyaKVB723uDpLw+w5e71eT9++fXn77bcBeP311+nTpw99+vTB09OT4cOHA7Bnzx78/f3p06cPgYGBstm4EKLcMZbgeJB27dpx+fJlkpKSyM3NJSYmBi8vL5My91u04uXlRUxMDLm5uSQlJXH58mWefvpps+osDbNH9oXpEgr3nf3222+V19555x169OgBgLu7Oz169ECj0XD+/HnCwsLYuXOnxQ0VQgi16A2lWptShI2NDVOmTGHw4MFK/rDmzZuzcOFC2rZtS48ePejfvz/h4eF4e3vj4ODA/PnzAWjevDm9evXC19cXa2trpkyZouzsd686LWVWiuOUlBTGjx/P0KFDWblyJcuWLVNey8rKonv37uzdu1dJoVDo5MmTTJw4kR07djywfklxLIQwlxopjg+69C++0P/3QspGi+9XHpg1si9Ml3Dz5s0ir+3Zswd3d3eTjn737t188sknpKenm/xiEEKI8sBI1Yv9sShdAsD27duL7DXr7e3Nzp07WbJkCQsXLlSnpUIIoRKD0fyjsrAoXUJ6ejpnzpyhW7du93xvx44dSUpKIj09XdVGCyGEJQxozD4qi2KnccaMGcOYMWMAOHbsGCtWrFDCfePi4ujWrRvVq1dXyv/+++80btwYjUbDuXPnyM3Nldw4QohypSpO41i0zj42Npa33nrL5FpcXBxbtmzBxsaGGjVqMH/+fMmNI4QoV/RVsLOXDceFEBWKGqtxdmpfNbvsS7p7p4apaCSCVghR5RgedQMeAbM6ey8vL2rXro2VlRXW1tZs2rSJHTt2sHjxYn777Tc2bNhAu3btlPLnz59n6tSpZGVlYWVlxcaNG03m9YUQ4lGSOfsHWLVqFfXq1VPOW7RowaJFi5g6dapJufz8fMLDw/n4449p2bIl165dw8ZGvkAIIcqPSrS1rNlK3Qu7ubnd8/rhw4d56qmnaNmyJYCsxBFClDuVaUmlucxOEDFo0CACAwNNUn3ey6VLl9BoNAwaNIiAgAC++OILixsphBBq0pfgqCzMGtmvXbsWrVZLWloaAwcOpFmzZnTs2PGeZfV6Pf/973/ZuHEjNWvWJDQ0lLZt2+Lu7q5qw4UQorQMVXA5uFkj+8Jcyk5OTnh7ez9w1xQXFxc6duxIvXr1qFmzJl27duXcuXPqtFYIIVSgVorjiqTYzj47O1tJa5ydnc3hw4cfmG7T09OTX3/9lZycHPLz8zlx4gRPPvmkei0WQggLGUpwVBbFTuOkpaUpm4nr9Xp69+5N165d2b17NzNmzCA9PZ23336bVq1a8eWXX+Lg4EBoaCj9+/dHo9HQtWvX++bOEUKIR6EqrsaRCFohRIWiRgTtNw0GmF12wJ/fWHy/8kAWwAshqpyqOLKXzl4IUeVUprl4c5U6XcKiRYtYv369ElX73nvv8eKLL5Kbm8vUqVM5e/YsGo2GSZMm0blz5zL9EEIIURIPa+76+vXrjB49mj/++IOGDRuyYMECHBwcipSLjo5m6dKlAAwbNoyAgABycnJ49913uXLlCtbW1nTv3l3ZS2TTpk3MnTtXWSk5YMAAgoKCHtiWUqdLAAgNDWXQoEEm1zZs2ADAtm3bSEtL46233mLjxo1YWamzwa8QQljqYU3jREZG4u7uzpAhQ4iMjCQyMpLw8HCTMtevX2fx4sV89913aDQaAgMD8fLywtbWljfffJMuXbqQm5tLaGgo+/fv58UXXwTA19eXKVOmmLh/UGcAACAASURBVN0W1XvgCxcuKCN5Jycn6tSpw9mzZ9W+jRBClNrDWnoZHx9P374FC1D69u3Lnj17ipQ5dOgQHh4eODo64uDggIeHBwcPHqRmzZp06dIFAFtbW1q3bo1Opyt1WyxKl7BmzRr8/f2ZMGECGRkZALRs2ZKEhATy8/NJSkri3LlzJCcnl7qBQgihNr3G/MMSaWlpODs7A/D444+TlpZWpIxOp8PFxUU512q1RTr1zMxM9u7da5KJYNeuXfj7+zNq1Ciz+thSp0t47bXXGD58OBqNhoULFzJnzhwiIiLo168fv/32G/369aNBgwa0b98ea2trc24jhBAPRUlG7FFRUSaD3ODgYIKDg5Xz0NBQ/v777yLvCwsLMznXaDSl2rUvPz+f9957j5CQEFxdXQHo3r07vXv3xtbWlnXr1jF+/HhWr179wHrM6uzvlS7hztw4QUFBDB06tKBCGxsmTpyovPbqq6/StGnTEn04IYQoSyXp7O/u3O+2cuXK+77m5OREamoqzs7OpKamFnnuCQX96/Hjx5VznU5Hp06dlPPJkyfTtGlTQkNDlWt3ZhMOCgri448/LvZzlDpdQmpqqlJmz549SgqFnJwcsrOzgYJ0x9bW1pIuQQhRrjys3DheXl5s3lwQBLZ582Z69OhRpIynpyeHDh0iIyODjIwMDh06hKenJwDz588nKyvLZAANmPS/CQkJ9005f6dSp0sIDw/n/PnzADRs2JDp06cr5QcNGoSVlRVarZa5c+cW2wghhHiYHtZqnCFDhhAWFsbGjRtp0KABCxYsAODMmTOsW7eOWbNm4ejoyPDhw+nfvz8AI0aMwNHRkZSUFP7zn//QrFkzAgICgP9bYvn111+TkJCAtbU1Dg4OREREFNsWSZcghKhQ1EiXML+x+ekSRl+RdAlCCFEhVaZNScxl1tLLzMxMRo0axUsvvUSvXr04efIk169fZ+DAgfTs2ZOBAwcqSy+3bt2Kv78//v7+vPrqq8pUjxBClBcGjflHZWFWZz9r1ixeeOEFdu7cyZYtW3Bzc1Miw3bt2oW7uzuRkZEANGrUiG+++YZt27YxbNgwJk+eXKYfQAghSqoq5rMvtrO/ceMGJ06cUB4e2NraYm9vf9/IsA4dOii5H5599llSUlLKqu1CCFEqVXGnqmLn7K9evUq9evWYMGEC58+fp02bNkyaNMmsyLCNGzfStWtX9VsthBAWMFSqbtw8xY7s8/Pz+emnn3jttdfYvHkzNWvWVKZsCt0rMuzo0aNs3LhRydImhBDlhb4ER2VRbGfv4uKCi4sLzzzzDAAvvfQSP/30kxIZBhSJDDt//jwffPABn3/+uUmklxBClAcyZ38Pjz/+OC4uLly8eBGAI0eO4Obmdt/IsD///JN33nmHuXPn8sQTT5Rh04UQonSq4mocs9bZT548mbFjx5KXl4erqysREREYDIZ7RoYtWbKE69ev8+GHHwIom50IIUR5URXn7CWCVghRoagRQTup6etml511+VuL71ceSAStEKLKqUxz8eaSzl4IUeXoq+A0TqnTJRRasWIFTz31FOnp6QAcO3aM5557jj59+tCnTx8WL15cNi0XQohSqoqrccwa2RemS/jss8/Izc3l1q1bACQnJ3P48GEaNGhgUv75559n2bJl6rdWCCFUUBUf0JY6XQJAREQE4eHhpdpqSwghHpWqmC6h2M7+znQJffv2ZdKkSWRnZ7Nnzx6cnZ1p2bJlkfecOnWKl19+mcGDB/O///2vTBouhBClVRWncUqVLmHRokUsW7aMd999t0j5Nm3akJCQwNatWwkJCVF2uRJCiPJCj9Hso7IodbqEq1ev0qdPH7y8vEhJSSEwMJC//voLOzs7ateuDcCLL75Ifn6+8vBWCCHKAwNGs4/KolTpElq3bs2RI0dISEggISEBFxcXNm3axOOPP85ff/1FYZxWYmIiBoNB8uMIIcqVhzVnf79Nnu4WHR1Nz5496dmzJ9HR0cr1kJAQfHx8lNWNhdmFc3NzCQsLw9vbm6CgIK5evVpsW0qdLuF+4uLiWLt2LdbW1tSoUYNPP/1UHuAKIcqVhzViL9zkaciQIURGRhIZGUl4eLhJmevXr7N48WK+++47NBoNgYGBeHl5KfuCzJs3j3bt2pm8Z8OGDdjb27N7925iYmKYN2+ekrLmfszq7Fu1avXA/DYJCQnKnwcMGMCAAeZv5iuEEA/bw3rwGh8fz9dffw0UbPIUEhJSpLM/dOgQHh4eODo6AuDh4cHBgwfp3bv3fetNSEhg5MiRAPj4+DB9+nSMRuMDB9ZmBVUJIURlYizB/yxhziZPOp0OFxcX5Vyr1aLT6ZTziRMn0qdPH5YsWaJMket0OurXrw+AjY0NderU4dq1aw9si1kj+8zMTD744AN+/fVXNBoNs2fPpkaNGkydOpXbt29jbW3NtGnTePrpp7lx4wbh4eH8+eef6PV63nzzTfr162fObYQQ4qEoySqbqKgooqKilPPg4GCCg4OV89DQUP7+++8i7wsLCzM5v9cmT8WZN28eWq2WrKwsRo0axZYtW5TtYEuq1BG0YWFhjBgxghdffJH9+/fz8ccf8/XXX7NmzRrc3Nz4z3/+Q3p6Oi+99BL+/v7Y2tqWqoFCCKG2kkzj3N25323lypX3fa1wkydnZ+cimzwV0mq1HD9+XDnX6XR06tRJeQ3Azs6O3r17k5iYSN++fdFqtSQnJ+Pi4kJ+fj43btwodiFMqSNoNRoNN2/eVMoUflUpvG40Grl58yYODg7Y2Ei+NSFE+WEwGs0+LHG/TZ7u5OnpyaFDh8jIyCAjI4NDhw7h6elpsmw9Ly+Pffv20bx5c6XewlU7cXFxdOnSpdhvDcXms//555+ZPHkyTz75pMmG48nJyQwaNAij0YjBYGDdunU0bNiQrKwshg0bxqVLl7h58ybz58+nW7duD2yE5LMXQphLjXz2A5oEml32m99Lv/nStWvXCAsLIzk5WdnkydHRkTNnzrBu3TpmzZoFwMaNG5V8YkOHDqVfv35kZ2czYMAA8vLyMBgMuLu7M2HCBKytrbl9+zbh4eH8/PPPODg4MH/+fFxdXR/YlmI7+zNnzhAcHMzatWt55plnmDlzJnZ2dmRlZdGxY0d8fHyIjY1l/fr1rFy5kp07d/Ljjz8yYcIErly5wsCBA9m6dSt2dnb3vYd09kIIc6nR2b/eJMDsst/+Hl18oQqg1BG0hUEAAL169SIxMRGATZs20bNnTzQaDU2aNKFRo0ZKQJYQQpQHD2s1TnlS6g3HnZ2dlYcKR48epWnTpgDUr1+fI0eOAPD3339z6dIlGjVqVEbNF0KIksvHaPZRWZQ6grZHjx7Mnj2b/Px8qlevzvTp0wEYPnw4EyZMwN/fH6PRyNixY+/5BFoIIR6VyjRiN5dsOC6EqFDUmLMPbPKy2WU3/b7V4vuVB7ImUghR5ZSDMe5DJ529EKLKqUypi81VbGd/8eJFRo8erZwnJSUxatQorl+/Tnx8PFZWVjg5OREREYFWq+W3335j4sSJnDt3jtGjRzNo0KAy/QBCCFFSlWlTEnOVaM5er9fTtWtX1q9fj4ODg7J2fvXq1Vy4cIHp06eTlpbGH3/8QXx8PPb29mZ19jJnL4Qwlxpz9r6Nfc0uG3sl1uL7lQclynp55MgRXF1dadiwoUmQVE5OjhKq6+TkxNNPPy0pEoQQ5ZbRaDT7qCxK1CPHxMSY5FieP38+mzdvpk6dOqxevVr1xgkhRFmoTBuJm8vskX1ubi4JCQm89NJLyrXRo0ezf/9+/P39+eabb8qkgUIIoTaJoH2AAwcO0KZNGx577LEir/n7+7Nr1y5VGyaEEGVFNhx/gJiYGPz8/JTzy5cvK3+Oj4+nWbNmqjZMCCHKit5oMPuoLMyas8/Ozub7779XUiIAfPLJJ1y6dAmNRkPDhg358MMPAfjrr7/o168fWVlZWFlZsWrVKmJjYx+Y9VIIIR6myjQ9Yy5JlyCEqFDUWHrZtWHRTUTu58Af8RbfrzyQ9ZFCiCrnkY9wH4FSR9CeOnWKS5cuAQXbEtapU4ctW7Yo5f7880/8/PwYOXKkRNEKIcqVyvTg1VzFdvbNmjVTOvHCCFpvb29CQ0OVMnPmzCkyJz9nzhxeeOEFdVsrhBAqkM6+GHdG0BYyGo3s2LGDVatWKdf27NlDw4YNqVWrlnotFUIIlTysVTbXr19n9OjR/PHHHzRs2JAFCxbg4OBQpFx0dDRLly4FYNiwYQQEBJCVlcW//vUvpUxKSgovv/wykyZNYtOmTcydOxetVgvAgAEDCAoKemBbLIqgBfjhhx9wcnJSdqq6efMmX3zxBStWrGDFihUlqV4IIR6Kh7UaJzIyEnd3d4YMGUJkZCSRkZGEh4eblLl+/TqLFy/mu+++Q6PREBgYiJeXFw4ODiZT44GBgcpWsAC+vr5MmTLF7LZYFEELsH37dpNfAIsXL+bf//43tWvXNrsRQgjxMD2s3Djx8fH07Vuw2rBv377s2bOnSJlDhw7h4eGBo6MjDg4OeHh4cPDgQZMyly5dIi0tjeeff77UbTF7ZH+vCNr8/Hx2797Npk2blGunT58mLi6OefPmkZmZiZWVFdWrV2fAgAGlbqQQQqjpYc3Zp6Wl4ezsDBTs552WllakjE6nw8XFRTnXarXodDqTMjExMfj6+ioJJwF27drFiRMneOKJJ5gwYQL169d/YFvM7uzvjqAF+P7772nWrJlJQ7/99lvlz4sWLaJWrVrS0QshypWSjNijoqKIiopSzoODgwkODlbOQ0ND+fvvv4u8LywszORco9GYdNYlERsby9y5c5Xz7t2707t3b2xtbVm3bh3jx48vNhllqSNoCxtw9y8AIYQo7/QlyHt5d+d+t5UrV973NScnJ1JTU3F2diY1NZV69eoVKaPVajl+/LhyrtPp6NSpk3J+/vx59Ho9bdu2Va7VrVtX+XNQUBAff/xxsZ/DrDn7WrVqcezYMerUqWNyfc6cObz22mv3fd8777wja+yFEOWOwWg0+7CEl5cXmzcXRPxu3ryZHj2KRu56enpy6NAhMjIyyMjI4NChQ3h6eiqvb9++vcigOjU1VflzQkICbm5uxbZFImiFEFXOw1qNM2TIEMLCwti4cSMNGjRgwYIFAJw5c4Z169Yxa9YsHB0dGT58OP379wdgxIgRODo6KnXs2LGDyMhIk3q//vprEhISsLa2xsHBgYiIiGLbIrlxhBAVihq5cVo5dyq+0P/3c+rx4gtVAGaN7FeuXMmGDRvQaDS0aNGCiIgINmzYwKpVq7hy5QpHjhxR5qKWL1/Otm3bgIKI299++40jR46Y/KYSQohHqSpmvSy2s9fpdKxevZrY2Fhq1KjBu+++S0xMDB06dKBbt2688cYbJuUHDx7M4MGDgYK5pJUrV0pHL4QoVyydi6+IzBrZ6/V6bt26hY2NDbdu3cLZ2ZnWrVsX+757RdwKIcSjVpk2JTFXsatxtFotb775Jt27d8fT0xM7OzuTJ8X3k5OTw8GDB03Ce4UQojyQPWjvISMjg/j4eOLj4zl48CA5OTkm+RruZ+/evXTo0EGmcIQQ5Y7RaDD7qCyK7ey///57GjVqRL169ahWrRo9e/bk5MmTxVZ8r4hbIYQoD2TD8Xto0KABp0+fJicnB6PRyJEjR4pdwH/jxg1OnDhxzwACIYR41B5WIrTypNjO/plnnsHHx4eAgAD8/f0xGAwEBwezevVqunbtapJjudDu3bvx8PCQfPZCiHKpKo7sJahKCFGhqBFUVd+x+NWEhZKv/2Tx/coDSZcghKhyKtMqG3OZlQht5cqV+Pn50bt3b9577z1u376N0Whk/vz5+Pj40KtXLyW9ptFoZObMmXh7e+Pv78+5c+fK9AMIIURJVcU5+1JH0BqNRpKTk9mxYwdWVlZKUv4DBw5w+fJldu3axenTp5k2bRobNmwo8w8ihBDmqkxz8eYya2RfGEGbn5+vRNCuXbuWESNGYGVVUIWTkxPwf9twaTQann32WTIzM03ScQohxKNWFUf2pY6gTUpKIjY2lsDAQAYPHszly5eBoltsubi4FNliSwghHiW9wWD2UVmUOoI2NzeX6tWrs2nTJl555RUmTpz4MNorhBAWq4pLL0sdQavVavH29gbA29ubX375BSj4JpCSkqK8PyUlBa1WW0bNF0KIkpNpnHu4XwTtP//5T44dOwbA8ePHadq0KfB/23AZjUZOnTpFnTp1lN3VhRCiPHhY2xKWJ8WuxrkzgtbGxoZWrVoRHBzMrVu3GDt2LKtWraJWrVrMmjULgBdffJH9+/fj7e1NzZo1mT17dpl/CCGEKImquM5eImiFEBWKGhG0NWs2MbtsTs7vFt+vPDBr6aUQQlQmBqPB7MMS169fZ+DAgfTs2ZOBAweSkZFxz3KDBg3i+eef5+233za5npSURFBQEN7e3oSFhZGbmwtAbm4uYWFheHt7ExQUxNWrV4tti3T2Qogq52E9oI2MjMTd3Z1du3bh7u5OZGTkPcsNHjyYuXPnFrk+b948QkND2b17N/b29mzcuBGADRs2YG9vz+7duwkNDWXevHnFtkU6eyFElfOwOvvCIFOAvn37smfPnnuWc3d3p3bt2kXaePToUXx8fAAICAggPj4eKNjfOyAgAAAfHx+OHDlSbFvLRSI0NebghBDCXHm5f5hdNioqiqioKOU8ODiY4OBgs96blpamrEZ8/PHHlbQy5rh27Rr29vbY2BR003cGqOp0OurXrw+AjY0NderU4dq1a9SrV+++9ZWLzl4IIcqr4jr30NBQ/v777yLXw8LCTM41Gg0ajUb19plLOnshhLDAypUr7/uak5MTqampODs7k5qa+sCR993q1q1LZmYm+fn52NjYmASoarVakpOTcXFxIT8/nxs3blC3bt0H1idz9kIIUUYKg0wBNm/eXKKtWjUaDZ07dyYuLg6A6OhovLy8lHqjo6MBiIuLo0uXLsV+aygX6+yFEKIyunbtGmFhYSQnJ9OgQQMWLFiAo6MjZ86cYd26dUow6uuvv87FixfJzs7G0dGRWbNm8cILL5CUlMTo0aPJyMigVatWzJs3D1tbW27fvk14eDg///wzDg4OzJ8/H1dX1we2RTp7IYSoAmQaRwghqgDp7IUQogqQzl6ICuS///2vWdeEuJt09qLK+euvv4iPjychIYG//vrrUTenRGbOnGnWNSHuVu7X2c+YMeOBS4o++OADi+pv3779A+v/8ccfLar/TleuXMHFxQVbW1uOHTvGL7/8Qt++fbG3t1ftHn///TeffvopqampLF++nAsXLnDy5EmCgoJUqX/Dhg0mden1epYuXcrIkSMtrrus2w4F7V+yZAldunTBaDQyc+ZMhg8fTv/+/VW7R25uLnFxcfzxxx/k5+cr1y35Ozp58iQnT54kPT2dr776SrmelZWFXq+3qL3388MPP/D777/Tr18/0tPTuXnzZrErPoqza9euB77es2dPi+oX91fuO/u2bdsCBZ3uhQsX8PX1BWDnzp24ublZXP/JkycBWLBgAY8//jh9+vQBYOvWraqP+t555x2+++47fv/9d6ZMmYKXlxdjxozhiy++UO0e77//PoGBgfznP/8BoGnTpowePVq1DvPo0aPs2rWLWbNmkZGRwfvvv0+nTp1Uqbus2w6wfPlyoqOjlQCUa9eu8eqrr6ra2Q8bNow6derQpk0bbG1tVakzLy+P7Oxs9Ho9N2/eVK7b2dnx2WefqXKPOy1evJizZ89y6dIl+vXrR15eHuHh4axbt86ievfu3QsUpBE4efIkXbp0AeDYsWO0b99eOvuyZKwggoKCjHl5ecp5bm6uMSgoSLX6/f39zbpmib59+xqNRqPxiy++MK5evdpoNBqNffr0UfUegYGBRep9+eWXVb1HTEyMsVOnTsZu3boZf/jhB9XqfRhtDw4ONt6+fVs5v337tjE4OFjVe/j5+ala352uXr1qNBqNxuzs7DK7h9FY8PduMBhMfha9e/dWrf6BAwcadTqdcq7T6YxvvvmmavWLoirMnH1GRgZZWVnKeXZ29n1zQ5dGrVq12Lp1K3q9HoPBwNatW6lVq5Zq9UNBwqLt27ezefNmunXrBmDyNV8NtWrV4tq1a8rUVOHWkGq5fPkyq1evxsfHhwYNGrBlyxZycnJUqbus2w7QuHFjXnnlFRYtWsTixYsJDg6madOmfPXVVybTI5Zo3769siez2lJTU/H19aVXr14AnD9/nmnTpql+n2rVqpnkcsnOzla1/uTkZJPtSh977DH+/PNPVe8hTJX7aZxCQ4YMISAggM6dO2M0Gjlx4gTvvPOOavXPmzePWbNmMWvWLDQaDR06dDArR3RJREREsG7dOoYOHYqrqytJSUm8/PLLqt7j/fffZ9iwYVy5coVXX32Va9eusXDhQtXqHzp0KFOmTOEf//gHRqORr776iv79+xMTE2Nx3WXddijo7Bs3bqycF4av3zk1Ulr+/v5AwXOMTZs20ahRI5NpnG3btll8j9mzZ/Pll18ybNgwAFq2bMkPP/xgcb1369WrF1OmTCEzM5P169fz3Xff8corr6hWv7u7O4MGDcLPzw+A2NhY/vGPf6hWvyiqQkXQ/vXXX5w+fRoo2Bv38ccff8QtKr2MjAySk5Np2bKl6nXn5+dz6dIljEYjTzzxBNWqVVOt7qysLOzs7EyuXbp0iSeeeEKV+suy7XfLyMjA3t5etUyEf/zx4LS5DRs2tPgeQUFBbNiwgb59+yo5V15++WW2bt1qcd2FjEYjKSkpXLx4kUOHDgHg6emJh4eHavcA2L17NydOnACgY8eOeHt7q1q/MFXuR/bnzp0zOS/M4Zyamkpqaipt2rRR5T6XLl1i2rRppKWlsX37ds6fP09CQgLDhw9XpX6AkJAQli5dSn5+PoGBgTg5OdGhQwcmTJig2j30ej379+/njz/+QK/Xc/jwYQAGDhyoSv23bt1i9uzZ6HQ6vvzyS2XFjBqd/d0rNS5fvkydOnVo0aIFTk5OFtW9ePFievXqhZubG7m5uQwePJjz589jbW3NJ598osqosrAzP3XqFE8++aTySzErK4vffvtNlc6+fv36/Pjjj2g0GvLy8li9erUqCxXupNFoGDJkCNu2bVO9g79T69atqV27Nv/4xz/Iycm550BCqKfcd/Zz5sy572sajYbVq1ercp/Jkyczbtw4pkyZAhR8PR47dqyqnf2NGzews7NTRmajRo1SvvqrZejQoVSvXp0WLVpgZaX+I5myXDGzceNGTp06RefOnQE4fvw4bdq04erVqwwfPlzZ8ac0duzYwYgRI4CC7IFGo5EjR45w+fJlxo8fr+oUwrRp05SMhFDwLOLua5bUPWvWLHQ6HV27dsXDw4OpU6daXO/dWrduTWJiIk8//bTqdQOsX7+eqKgoMjIy2LNnDzqdjqlTp7Jq1aoyuZ+oAJ39119/jcFg4OTJkzz33HNldp+cnJwi/2FbW1ureg+9Xk9qaio7duwosrGBWlJSUlSZG76fa9eu4evrq+ylaWNjo9ovFb1eT2xsLI899hhQsO5+/PjxrF+/ngEDBljU2Rc+cAQ4dOgQfn5+WFtb4+bmpvo6daPRaDI1ZGVlpdqD+DNnzvDJJ5+YXFu7di2vvfaaKvUXOn36NNu2baNBgwbUrFlTua7Wf1tr1qxhw4YNynOApk2bkp6erkrd4t7KfWcPBf9YZsyYocxRloW6dety5coV5R/pzp07VX8mMHz4cAYNGsRzzz3H008/TVJSEk2bNlX1Hl27duXQoUN4enqqWm+hslwxk5ycrHT0ULDxQ3JyMo6OjsrWbKVla2vLr7/+ymOPPcaxY8cYN26c8ppaq4kKubq6snr1aqUD/vbbby0ORiq0dOlSbG1tcXd3BwriBo4ePap6Z//ll1+qWt/dbG1tTR5eq70qTRRVITp7KHh6HxcXR8+ePctka6+pU6cyefJkLl68yAsvvECjRo1UX43Tq1cvZckcFHQKixYtUvUezz77LCNHjsRgMGBjY6OMMtWKBC7LFTOdOnXi7bff5qWXXgIKNmXo1KkT2dnZFv9CmTRpEqNGjeLatWv8+9//Vjrf/fv307p1a4vbfqcPP/yQmTNnsnTpUjQaDe7u7syYMUOVuj///HOGDh1KtWrVOHjwIBcvXuTzzz9Xpe47FT5fSEtL4/bt26rX37FjR/7zn/9w69YtDh8+zLfffqtszCHKRoVZjdO+fXtycnKwtramevXqqndier0ea2trsrOzMRgMZfKg6Pbt22zcuJH//e9/Jv+AIiIiVLuHl5cXn3/+OU899ZSqvxQTExOpX78+jz/+OPn5+URFRREXF8eTTz7JqFGjcHR0tPgeRqORXbt2KYm97O3tSUtLK5M56bKi1+sZN25ckakWNaWlpREaGkrbtm2ZPXt2mQx+4uPj+eijj5St9P7880/c3NxUWWILYDAY2Lhxo8lqHzWXdoqiKszIvjCtQVnp0aMHL7zwAr6+vkoIt9rCw8Np1qwZhw4dYsSIEWzbto1mzZqpeo/69evTokUL1TuAqVOnKkFHJ0+eZOnSpUyePJmff/6ZKVOmqBKyr9FocHV15dSpU8TFxdGwYUN8fHwsrvdO165dY8mSJfz3v/9V4ilGjBhR7P6d5rK2tubPP/8kNzdXtVQJ8H85nAoHOXl5eVy9epWdO3eqOugptHDhQqKiohg4cCCbN2/m6NGjqi7vXLRoEe+++67Swev1esaMGVOmvySrugrT2RuNRrZu3crVq1cZMWIEycnJ/PXXX6qtFtixYwd79+5lzZo1TJo0iW7duuHr68vzzz+vSv1QkAjts88+Iz4+noCAAHr37s2//vUv1eqHgqmhkJAQunbtatLZWLr0Uq/XK6P32NhYgoOD8fHxwcfHR8knVFqXb+ZJqgAADTVJREFULl0iJiaG7du3U7duXXx9fTEajXz99dcW1Xsv7733Hs8//7zyy2nbtm2MHj36gZtGl5SrqyuvvfYaXl5eJlHYlvwMynqwczcbGxvq1q2LwWDAYDDQpUsXZs+erVr9KSkpLFu2jLfffpvc3FzCwsJo1aqVavWLoipMuoRp06Zx6tQptm/fDhQ8KPzwww9Vq79mzZr4+vqyePFioqOjycrKIiQkRLX6AeUho729Pb/++is3btwgLS1N1Xs0atQId3d38vLyuHnzpnJYymAwKA/Rjhw5YvLtx9LVLL169eLo0aMsW7aMtWvXEhISUibLRqEgMG/EiBG4urri6urK8OHDVf8ZNG7cmO7du2M0GlX9GUBBINKNGzeU88zMTPbs2aNK3Xeyt7fn5s2bdOzYkbFjxzJz5kxV04fMnj2bX3/9lWXLljF06FA6deqkakS8KKrCjOwTExOJjo5Wlt85ODiQl5en6j2OHz9ObGwsBw8epG3btixYsEDV+oODg8nIyODdd99l2LBhZGdnM2rUKFXvoUaq4Xvx8/NjwIAB1K1blxo1aijfeH7//XeLn28sXryYmJgY3njjDV544QX8/Pwoq0dJHh4exMTEKA/Kd+7cqfrKpbL6GUDB39Wdkab29vYsXryYf/7zn6rU/+eff9KgQQM+//xzatSowYQJE9i2bRs3btxQ4hQscWeQ5BtvvMGUKVPo0KEDHTt25Ny5c6oFSYqiKswD2qCgINatW0f//v2Jjo4mPT2dN998U7XlmF5eXrRq1YpevXoV+fpdkaSnp/PFF19w4cIFk4fAagSfnTp1ir/++gsPDw/l7+fSpUtkZ2er8o80Ozub+Ph4YmJiOHr0KH369MHb21uVzvjOOe/CB/1Q8K2kVq1aqs55l+XPwN/fv8ha93tdK62AgAAl+Oudd95RfbXYg74tqxkkKYqqMCP7kJAQRowYQVpaGvPnz2fnzp2qBiZt3bq1zEK1i8umqFYqA4CxY8fSq1cv9u3bx4cffkh0dDT16tVTpe5nn322yDW1cuJAwdScv78//v7+ZGRksHPnTr744gtVOvuHOeddlj+Dtm3bEhERoTzrWbNmjaqj4TvHfklJSarVW6gwSHLnzp3K3hTi4agwnf3LL79MmzZtOHr0KEajkc8//1yVnCBffPEFb731FvPnz7/nChZLd8ICdTIqmuv69esEBQWxevVqOnXqRKdOnejXr99Du79aHBwcCA4OJjg4WJX6fvvtN9zc3IrkWiqkZodZlj+DyZMn8/nnnysDHQ8PDyXFhxru/DdQFks6oSBIcvny5dLZP2QVprO/fv06Tk5OSkpUKNi9x9KsiIW/MAp3xCoLZTmHe7fCh8DOzs7s27cPZ2dnVfP+V1QrV65kxowZJrmW7uzM1Jw+KMufQa1atRg7dqwqdd3L+fPn6dChA0ajkdu3b9OhQwcA1eNa/vGPf/Dll1/i6+trko5BjXgNcW8VZs7ey8uL5ORkZb/WzMxMHnvsMR577DFmzJhhcWf9MB4OjR8/nkmTJimfISMjgzlz5qgaVLV3716ef/55kpOTmTFjBjdv3mTEiBFK3vaq6s6gMChIhhYXF0ejRo0YOXKkqp1MWfwMpk+fzpQpUxg6dOg9Xy9MTFdR3CtaVqPREB8f/whaUzVUmM7+gw8+wMfHhxdeeAEoSGa1a9cuAgMDmTVrFhs2bLCo/pCQEP7++298fHzw9fWlRYsWajTbxJ05yB90TagvICCAr776CkdHR06cOMHo0aOVoLCLFy+qEhR2+/Zt1q5dy5UrV2jRogX9+/e3OKdPoQ4dOvDjjz9y/Pjxe76u1j7AovKqMNM4p0+fZubMmcq5p6cn/6+9uwuJaoviAP6fPGnmQyGpkKIQFYqKlEMagSl9kAYhMiMmZaBPUVBQYSaJBCFCRZgVfpBSQQkVyJSSRYURWlBSFEnNZExmhGFjjJai432QOdfj1XtvzZ6PPfP/gTB4YLcf5ixP66y9VnV1NU6cOIHx8XGX179y5QoGBwfR3t6OiooKjIyMIDs7W2iLY4fDgeHhYSxZsgTAdGpKVMfF2traea/pdDohZXMyc+ehMKfS0lIoigK9Xo/Ozk6YzWYh73wAqNO1/Cmov3v3DmazWXP/utLZlP6dNME+IiIC9fX1mjFmy5Ytw+TkpLADOBERESgqKkJaWhoaGxtx4cIFocG+uLgY+fn5mhrv+f5b/rvmKhUdHR3FzZs3YbPZAj7YOw+FKYqCrq4uTWMyUX9wLRaLWgJpMBiE9Ph3Ghoa+teqLpEVXZ5QW1uLp0+fwmKxYOPGjejs7ERqaiqDvRtJE+xPnTqF8+fPq0Fr7dq1OH36NCYnJ4UcfrJYLGhra0NHRweWLl2K7OxsHD161OV1Z8rNzUVSUhK6u7sBTH/hV65cKWTt4uJi9bPdbsfly5dx69Yt5OTkaK4FKnceCnOambIRlb5xcjgcHq3qcre7d++itbUVubm5qKqqwrdv33DkyBFvb8uvSRPsw8PDcfz48TmvxcXFubz+sWPHkJOTg8bGRkRFRbm83kyzc7kFBQXCgwEwnRZqamqCyWRSD8c4U0aBbu/evVi/fr16KMxZieNwOOb9Xv0uZyULAE01i4hKloiICI9WdblbSEgIFixYAEVRYLfb1dkF5D7SBPu+vj5cunQJnz9/1gw6EFEyNzk5iZiYGOzZs8flteYyO5drsVhQXl4u9N+orq7GvXv3kJ+fD5PJhLCwMKHr+wN3Hwp7+/atsLVmk6SO4n9LSkrCjx8/YDQakZeXh8WLF2PNmjXe3pZfk6YaZ8eOHSgoKEBSUpImRy+qPr6wsBDNzc1C29I6zTzOPjExAaPRKGQe6Uzx8fEIDg5GUFCQpn5cdH00eYfNZvPbGvT+/n7Y7XbEx8d7eyt+TZone0VRUFhY6Lb1Y2JihLeldXJnLtept7fXLeuSb/DHQO8cVKPT6ZCamspg72bSPNmfO3cO4eHh2LJli+bpW9RNMF/poog8aUJCgnpK0JnLXbRoEZ+6KWBVVlbCarVqqutiY2OlmkomG2mCPU/cEfmPbdu2ob29XfOifPv27Whvb/fyzvyXNGmcBw8euHX93bt3z9n4iS1XicSLi4vDwMCAOtj8y5cvQqrqaH7SBPufP3+iqalJ7Tfy8eNH9PX1ISsrS8j6paWl6uexsTF0dHSoPc+JSAznIcKRkRHk5OSoY0VfvXolbMQozU2aYF9WVobExES1L3lUVBQOHDggLNjPrupJTU2FwWAQsjYRTeMBP++RJthbrVacPXsWd+7cATA9M1bk6wabzaZ+djgceP36tWbWJxG5bnZvH7vdrjk3Q+4jTbAPDg7Gr1+/1Ly61WoVWhOfl5enrq0oCqKjo3Hy5Elh6xPR31paWlBTU4OQkBB1XCQLLtxLmmqcJ0+e4OLFizCbzdiwYQN6enpQVVWFtLQ0l9b1ZJ9zIpq2detWXL9+Xdi4Rvpv0gR7APj+/TtevnyJqakppKSkCPmieKLPORFplZSUoLa2VjOlitxLmjTO8+fPkZCQgMzMTLS2tqKurg5FRUVq6daf8kSfcyLSOnToEAoKCpCSkqJJx4rq/0//JKYRvAdUVlYiNDQUvb29aG5uRmxsrKZc8k85+5wDQFdXF9LT09VrovqcE5FWRUUF0tPTkZKSgsTERPWH3EeaJ3tFUaDT6XD//n0UFhbCaDTixo0bLq/riT7nRKQ1MTGBsrIyb28joEgT7MPCwlBXVweTyYSrV69qnshd4Yk+50SklZGRgZaWFmRlZbml1xX9kzQvaAcHB3H79m0kJydDr9djYGAAz5494xgzIgmx15XnSRPsR0dHERISgqCgIPT19eHDhw/IyMjAwoULvb01IiKfJ80L2l27dmF8fBxfv35FSUkJWltbhc+IJSL3amhoUD/P7nB55swZT28noEgT7KemphAaGoqOjg7s3LkTNTU1eP/+vbe3RUS/oa2tTf1cX1+vufb48WNPbyegSBXse3p6YDKZkJmZqf6OiOQx856dff/yfnYvaYJ9eXk56urqsHnzZqxatQqfPn1yuVUCEXnWzJkRs+dHzDVPgsSR5gUtEcnPOaJz5nhOYPqpfnx8HG/evPHyDv2XNMF+aGgIDQ0NMJvNGBsbU3/PSVJERP9NmjTO4cOHsWLFCvT392P//v2Ijo5GcnKyt7dFRCQFaYK9zWaD0WiEoihYt24dqqqq0N3d7e1tERFJQZp2CYoyvdXIyEg8evQIkZGRGB4e9vKuiIjkIE3O/uHDh9Dr9erA8ZGREezbtw+bNm3y9taIiHyezwf7sbExXLt2DVarFatXr4bBYFCf8omI6P/x+WB/8OBBKIoCvV6Pzs5OLF++nAMOiIh+k88/IlssFphMJgCAwWCA0Wj08o6IiOTj89U4M1M2TN8QEf0Zn0/jOE/cAdCcupuamoJOp8OLFy+8vEMiIt/n88GeiIhc5/NpHCIich2DPRFRAGCwJyIKAAz2REQBgMGeiCgA/AXaXvjWUkMJugAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Missing Values handled successfully!\n",
        "Now we will convert features"
      ],
      "metadata": {
        "id": "jnDEWp76Ow1p"
      },
      "id": "jnDEWp76Ow1p"
    },
    {
      "cell_type": "code",
      "source": [
        "test_df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uK1GgPasY-JP",
        "outputId": "9fb35718-8984-4382-e923-117fce23fa17"
      },
      "id": "uK1GgPasY-JP",
      "execution_count": 248,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 418 entries, 0 to 417\n",
            "Data columns (total 10 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  418 non-null    int64  \n",
            " 1   Pclass       418 non-null    int64  \n",
            " 2   Name         418 non-null    object \n",
            " 3   Sex          418 non-null    int64  \n",
            " 4   Age          418 non-null    int64  \n",
            " 5   SibSp        418 non-null    int64  \n",
            " 6   Parch        418 non-null    int64  \n",
            " 7   Ticket       418 non-null    object \n",
            " 8   Fare         417 non-null    float64\n",
            " 9   Embarked     418 non-null    int64  \n",
            "dtypes: float64(1), int64(7), object(2)\n",
            "memory usage: 32.8+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert Fare feature to integer and fill NaN with 0\n",
        "data = [train_df,test_df]\n",
        "for dataset in data:\n",
        "    dataset['Fare'] = dataset['Fare'].fillna(0)\n",
        "    dataset['Fare'] = dataset['Fare'].astype(int)"
      ],
      "metadata": {
        "id": "eLkb1DajQAG2"
      },
      "id": "eLkb1DajQAG2",
      "execution_count": 250,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 251,
      "id": "db091ef8",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "db091ef8",
        "outputId": "7fc6b249-2ace-4951-f55e-6882b19b6eef"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "sns.heatmap(train_df[[\"Age\",\"Sex\",\"SibSp\",\"Parch\",\"Pclass\",'Embarked']].corr(), annot = True)\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Dropping name and ticket feature, as it is of less importance\n",
        "\n",
        "train_df = train_df.drop(['Ticket','Name'],axis=1)\n",
        "test_df = test_df.drop(['Ticket','Name'],axis=1)    "
      ],
      "metadata": {
        "id": "nuXiAg7WfwAo"
      },
      "id": "nuXiAg7WfwAo",
      "execution_count": 252,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Looks good now\n",
        "train_df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MEzwc4ODfv6N",
        "outputId": "ca5891cf-d56b-4f60-d4b7-d06151458254"
      },
      "id": "MEzwc4ODfv6N",
      "execution_count": 253,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 9 columns):\n",
            " #   Column       Non-Null Count  Dtype\n",
            "---  ------       --------------  -----\n",
            " 0   PassengerId  891 non-null    int64\n",
            " 1   Survived     891 non-null    int64\n",
            " 2   Pclass       891 non-null    int64\n",
            " 3   Sex          891 non-null    int64\n",
            " 4   Age          891 non-null    int64\n",
            " 5   SibSp        891 non-null    int64\n",
            " 6   Parch        891 non-null    int64\n",
            " 7   Fare         891 non-null    int64\n",
            " 8   Embarked     891 non-null    int64\n",
            "dtypes: int64(9)\n",
            "memory usage: 62.8 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.head(5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "cNESVGD2fvqS",
        "outputId": "3dd4bfee-6000-4bf3-d264-1b5d7feb6e93"
      },
      "id": "cNESVGD2fvqS",
      "execution_count": 254,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  Sex  Age  SibSp  Parch  Fare  Embarked\n",
              "0            1         0       3    1   22      1      0     7         2\n",
              "1            2         1       1    0   38      1      0    71         0\n",
              "2            3         1       3    0   26      0      0     7         2\n",
              "3            4         1       1    0   35      1      0    53         2\n",
              "4            5         0       3    1   35      0      0     8         2"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b59d1fd4-9e3c-417d-a4b0-bf4edb49a7ce\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>22</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>38</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>26</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>35</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>35</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b59d1fd4-9e3c-417d-a4b0-bf4edb49a7ce')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b59d1fd4-9e3c-417d-a4b0-bf4edb49a7ce button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b59d1fd4-9e3c-417d-a4b0-bf4edb49a7ce');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 254
        }
      ]
    },
    {
      "cell_type": "markdown",
      "id": "db7327e5",
      "metadata": {
        "id": "db7327e5"
      },
      "source": [
        "# FEATURE IMPORTANCE "
      ]
    },
    {
      "cell_type": "markdown",
      "id": "f0b2c149",
      "metadata": {
        "id": "f0b2c149"
      },
      "source": [
        "print (train_df.corr()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 255,
      "id": "4f7d05c7",
      "metadata": {
        "id": "4f7d05c7",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "outputId": "efbd82c3-eb6e-467b-970f-d43c4b1cfe2e"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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facffvhBknTkyBG999576tKlS51lunTpor1792r37t2SpMLCwoDPaQregzLM/fffr82bN+vgwYO66aabdN9992nFihX68ccf5fV6NWbMGEVFRalfv36aOHGiioqKOEkiBH344YdasWKFunXrpkGDBkk6vu3rM3LkSE2bNk0DBgzQJZdcop49ewZqVDSib775RlOnTpXH45HX69Utt9yifv366bHHHvMt07RpU+Xm5mr8+PG+kyTC9T9GudQRAMBIHOIDABiJQAEAjESgAABGIlAAACMRKACAkTjNHLDRpZdeqm7dusnj8ahLly566hzxca0AAAJlSURBVKmn6v2FzOeff57LVwE/wx4UYKNmzZppxYoVWr16tc477zwtXrw42CMBIYM9KCBA+vTp47sCxFtvvaVXXnlFDodD3bt31zPPPFNn2ddff11LlizR0aNHFRcXp6efflrNmzfX2rVr9eKLL6pJkyZq3bq1XnvtNe3cuVPTpk3T0aNHVVtbq+eff16dO3cOwisEGheBAgLg2LFjKi4u1o033qidO3fqpZde0qJFi9S+fXtVVFScsHxqaqqGDx8uSXr22We1bNkyjR49WrNmzdIrr7wiy7J8l8xZvHixxowZo1tvvVU1NTWqra0N6GsD7EKgABsdOXLEdymjPn36KCsrS0uWLNEtt9yi9u3bSzr+MQv/186dO/Xcc8+psrJSVVVVuuGGGyRJvXv31tSpUzVgwADfVc+vuOIKzZ49W+Xl5erfvz97TzhnECjARj+9B9VQU6dO1axZs9SjRw+9+eab2rx5syQpNzdXn3zyid555x0NHTpUb7zxhjIzM9WrVy+98847Gj9+vKZPn67rrruusV8KEHCcJAEEWFJSktatW6eDBw9K0kkP8VVVVSk6OlpHjx7VqlWrfI/v3r1bvXr1UnZ2ttq1a6fy8nKVlZUpNjZWY8aMUUpKClc6xzmDPSggwOLj43XPPfdo9OjRatKkiS677DI9+eSTdZbJzs7WsGHD1L59e/Xq1ct3Neunn35aX3/9tbxer5KSktSjRw+9/PLLWrFihSIiInTBBRfo7rvvDsbLAhodVzMHABiJQ3wAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjPT/ATEtxWWfD8g5AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "#PCLASS\n",
        "# Pclass is contributing to a persons chance of survival, especially if this person is in class 1. \n",
        "\n",
        "g = sns.catplot(x=\"Pclass\", y=\"Survived\", data=train_df, height=6,\n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "\n",
        "g.set_xticklabels([\"1st\", \"2nd\", \"3rd\"])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 256,
      "id": "fdae017e",
      "metadata": {
        "id": "fdae017e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "outputId": "d854012d-999e-464c-a284-3b47f1b2abb7"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 474.375x432 with 1 Axes>"
            ],
            "image/png": "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VD6DOnPmTLVq1Ur33HOPpO/O8fvII49o4cKFl/3gbrdbX3zxhVatWqWCggLdc8892rx5s8LCwi5wH4+Ki8su+7ElKSKijZF1YI6pv1sAgaUp/HvsM6iHDh3Sli1bvNfj4uKUlJTkc2GHw6GCggLv9cLCQjkcjnO26dWrl5o1a6aoqChdf/31ysvL009/+tOLeQ4AAPidz69v+8lPfqIPP/zQe/2f//ynYmNjfS7cs2dP5eXl6ejRo3K5XMrOzpbT6ay1za233qq9e/dKkk6dOqW8vDzva64AAASSOvdQk5OTJUnV1dUaPXq0rrnmGknS119/rU6dOvleODhYaWlpmjBhgtxut0aOHKmYmBhlZGQoNjZWCQkJGjBggN59910lJSXJbrdr5syZat++vaGnBgBAw7F5zncqJElfffXVBe947bXXWjKQL1VVbqOvoY5J22FkLX9anT5IB5+7z99jXLauD6/QiRMllq2/f/8+bd68UcnJI3TjjX0texwAF69Jv4b6w2CePHlSlZWVlg8EWGXdutX6/PMjqqgoJ6gAjPP5pqScnBw988wzOn78uMLDw/X111+rc+fOys7Oboj5AGPKyytqXQKAST7flJSRkaG1a9fq+uuv1/bt27VixQr16tWrIWYDACBg+AxqcHCw2rdvr5qaGtXU1CguLk4HDhxoiNkAAAgYPg/5hoWFqbS0VH379tXDDz+s8PBwtWzJ2WwAAPg+n3uoixYtUosWLfT4449rwIAB6tixoxYvXtwQswEAEDB87qG2bNlSJ06cUG5urtq2batbbrmFz4oCAPADPvdQ161bp1GjRuntt9/WW2+9pbvuukvr169viNkAAAgYPvdQly1bpo0bN3r3SouKijR69GilpqZaPhwAAIHC5x5q+/bt1apVK+/1Vq1accgXAIAfqHMPdfny5ZKkjh076s4771RCQoJsNptycnLUrVu3BhsQAABJ6tGjh7p27Sq3261OnTrpmWeeUWho6Hm3femll9SyZUvdf//9DTZfnXuopaWlKi0tVceOHXXrrbfKZrNJkhISEnTdddc12IAAAEhSixYtlJmZqaysLDVr1kyvv/66v0eqpc491ClTptS6XlpaKkm1Dv8CAOAPffv21aeffipJ2rRpk1555RXZbDZ169ZNzz77bK1t33jjDa1du1ZVVVWKjo7W/PnzFRoaqq1b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7fqv//ovSVJlZaWOHTumzp07N8j8aPwI6hXI4/EoJiZGa9euPe/tISEhkqSgoCDvn89er66u1p49e/Tee+9p7dq1Cg0N1dixY1VZWXnOY4wYMUL/8R//Yd0TAS6S0+nU/PnztXLlShUXF3t/npGRoX79+unll19Wfn6+xo0bd977L1y48JxDwcBZHPK9Av34xz/WqVOn9MEHH0iSqqqqdOjQoXrfv6SkRG3btlVoaKgOHz6sDz/88Jxt+vfvr7feeksnT56UJBUXF+urr74y8wSAS5SamqoHH3xQ3bp1q/XzkpIS75uUzp4O9YduueUWvfrqq96vmPz444+tHRYBh6BegUJCQrRw4UI999xzuv322zV8+HBvXOtj4MCBqq6uVmJiop5//nn17t37nG26dOmi6dOna/z48UpOTtb48eN14sQJk08DuGiRkZHn3fucMGGCFixYoOHDh6u6uvq89508ebKqq6t1++23a9iwYcrIyLB6XAQYTj0IAIAB7KECAGAAQQUAwACCCgCAAQQVAAADCCoAAAZwYgfAjxYvXqysrCwFBQUpKChI6enptU6JByBwEFTATz744APt2LFDGzduVEhIiE6dOnXeU94BCAwc8gX85MSJE2rfvr339I7h4eFyOBw6cOCA7rnnHqWkpOj+++/X8ePHVVJSoqFDh+rIkSOSpBkzZuiNN97w5/gAfoATOwB+UlpaqjFjxqiiokL9+/dXUlKS+vTpo7Fjx2rRokUKDw/Xli1b9M477+jpp5/Wu+++q4ULF2rcuHHasGGDXnnlFX8/BQDfQ1ABP3K73dq3b5/27NmjtWvXatKkSVqwYIGioqIk/f83AZ39hpPf//732rZtmzIzMxUZGenP0QH8AK+hAn5kt9vVr18/9evXT127dtVrr71W5zcB1dTU6PDhw2rRooW+/fZbggo0MryGCvjJkSNHlJeX573+ySefqHPnznV+E9CKFSvUuXNnPf/883rsscd4AxPQyLCHCvhJWVmZ5syZo9OnT8tutys6Olrp6em66667NGfOHJWUlMjtduvee++V3W7XunXrtG7dOrVu3Vo333yzFi9erKlTp/r7aQD4F15DBQDAAA75AgBgAEEFAMAAggoAgAEEFQAAAwgqAAAGEFQAAAwgqAAAGPB/EocrNhfkYf8AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "#SEX with Pclass\n",
        "# Shows that in general females have higher probability of survival than men irrespective of the Pclass\n",
        "g = sns.catplot(x=\"Sex\", y=\"Survived\", hue = \"Pclass\", data=train_df, height=6, \n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "\n",
        "g.set_xticklabels([\"Female\", \"Male\"])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))\n",
        "f = train_df[train_df['Sex']==0]\n",
        "m = train_df[train_df['Sex']==1]\n",
        "ax = sns.histplot(x=\"Age\", hue=\"Survived\",  data=f,ax = axes[0],bins=30, stat = 'percent')\n",
        "ax.set_title('Female')\n",
        "ax = sns.histplot(x=\"Age\", hue=\"Survived\",  data=m, ax = axes[1],bins=40, stat = 'percent')\n",
        "ax.set_title('Male')\n",
        "# 0 - Not survived; 1 - Survived\n",
        "# Men have higher probability of survival for age group 18-35 years old\n",
        "# Women have higher probability of survival for age group 16-34 years old (Ignoring the imputed valueof 25 years old)\n",
        "# In general infants (<10 years) have high probability of survival irrespective of gender"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 313
        },
        "id": "hO7wYBfbkGfV",
        "outputId": "14315331-23db-4d3b-e1c8-88d6d7a018e6"
      },
      "id": "hO7wYBfbkGfV",
      "execution_count": 257,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Male')"
            ]
          },
          "metadata": {},
          "execution_count": 257
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 260,
      "id": "20042d66",
      "metadata": {
        "id": "20042d66",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 865
        },
        "outputId": "19892f65-17e1-43d0-c1ee-149bc96be006"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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eXiZ+de43W6VlZWppKQk+JLYaP+hk6S7775bmzZt0oEDBzR27Fj96Ec/0rRp0yI9Vqe89957WrVqlYYPH67i4mJJXx1nbm5uhCf79j7//HPdd999sm1bgUBAV111lcaPHx/psXAS+/bt0x133CHpq+cNJ02apLFjx0Z4qs4ZOXKkCgoKNHnyZLndbp177rm69tprIzYPlzoCABiJh/gAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFdJFFixbJ6/WqqKhIxcXF+uCDD/TAAw9ox44dkqQLL7zwpF/3/vvva9q0aSouLtbEiRP19NNPd+XYQMTwd1BAF9i6das2bNiglStXKj4+Xvv379exY8c0f/78037tvffeq4qKCo0YMUK2bWvnzp1dMDEQeZxBAV1g7969SklJUXx8vCQpNTVVlmVpxowZ+vDDD4P7Pfroo/J6vZo5c6b2798v6aurS6SlpUn66jp9w4YNkyQ9/fTTuueee3TttddqwoQJevXVV7v4qABnESigC1x++eXas2ePCgoK9PDDD2vTpk0n7NPY2Kjs7GxVVVXp0ksv1a9+9StJ0syZM3XVVVfpjjvu0CuvvKKWlpbg1/zzn//Uc889p1deeUW//vWvufI+YgqBArpAnz59tGLFCpWXlys1NVU//vGPT3j31R49eqiwsFCSVFxcrPfee0+SdOedd+q1117T5ZdfrrVr16qkpCT4NXl5eerVq5dSU1M1atSodmdjQLTjOSigi7hcLo0aNUqjRo3S8OHD9ac//anD/f/3jQoHDx6s66+/Xtdcc40uu+wyHThw4IR9gFjDGRTQBT755BPV19cHb2/fvl0DBw5st09bW5veeOMNSdKaNWt08cUXS5I2bNig/14yc9euXerRo4eSk5MlSdXV1WppadGBAwe0adMm5eTkdMHRAF2DMyigCzQ2NupnP/uZvvzyS7lcLmVmZqq8vFylpaXBfXr37i2fz6dFixYpNTVVTz31lCRp1apVWrBggXr16iWXy6WFCxfK5XJJks455xzdeOONOnDggG6//XbeuwwxhauZA1Hq6aefVu/evXXzzTdHehTAETzEBwAwEmdQAAAjcQYFADASgQIAGIlAAQCMRKAAAEYiUAAAI/0/O8f1UAmZzCYAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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zp0hnipw/vGPf3gnTZrkHTNmTKhHCYjrrrvOe+TIkVCPEVDZ2dnel156yev1er0ul8t77NixkM3CKb4fqMslmhqiK6+8Uq1btw71GAHVvn17de/eXZIUFRWlzp07y+l0hniq+rHZbGrVqpWkUz8bWFVV1Wh+9KKkpERvvfWWsrKyQj0KalFWVqb33nvP9z2KiIhQTExMyOYhUD9wpks0NfS/9JqCgwcP6qOPPlKPHj1CPUq9eTweDRkyRH369FGfPn0axT5J0pw5czRlyhQ1a9a4/tq5++67lZmZqRdffDHUo9TbwYMHFRsbq2nTpmno0KG67777VFlZGbJ5GtfvFDRJFRUVGj9+vKZPn66oqKhQj1NvYWFhWrt2rTZt2qSioiLt2bMn1CPV25tvvqnY2FglJSWFepSAWrFihVavXq1nn31Wzz//vN57771Qj1QvVVVV2r17t0aNGqU1a9YoMjLyjB+VFCwE6gfqcokmmOPkyZMaP368MjIy1L9//1CPE1AxMTG6+uqrtXnz5lCPUm87duxQQUGBHA6HJk2apHfffVeTJ08O9Vj19t3fDe3atVNaWtppF8RuaOLj4xUfH+87ar/hhhu0e/fukM1DoH6gLpdoghm8Xq/uu+8+de7cWXfeeWeoxwmIb775Rt9++60k6cSJE3rnnXfUuXPnEE9Vf3/4wx9UWFiogoICPfHEE+rdu3eD//y3yspKlZeX+3799ttvKzExMcRT1U9cXJzi4+P16aefSpK2bNmiLl26hGwe3mb+A+Hh4crJydHo0aN9b4lt6L/pJGnSpEnatm2bSktL1bdvX917770aMWJEqMeql/fff19r165V165dNWTIEEmn9jMlJSXEk/10hw4d0tSpU+XxeOT1enXDDTfouuuuC/VYOIMjR47od7/7naRTrxsOHjxYffv2DfFU9Xf//fdr8uTJOnnypBISEvTII4+EbBYudQQAMBKn+AAARiJQAAAjESgAgJEIFADASAQKAGAkAgUEQbdu3TRkyBANHjxY48eP1/Hjx+u1vYMHD2rw4MEBmg4wE4ECgqBFixZau3at1q1bp3POOUcvvPBCnb6uqqrK4skAc/GDukCQ9erVSx9//LEKCgo0f/58nTx5Um3atNHjjz+uc889V3/961+1f/9+HThwQOeff76mT5+uBx54QAcOHJAkPfjgg2rfvr08Ho9mzJihnTt3ym63a968eWrRokWI9w4IHI6ggCCqqqpSYWGhunbtqiuuuEIvvfSS1qxZo0GDBmnRokW+9fbt26clS5boiSee0OzZs3XllVfq1Vdf1erVq31XNvn888918803Ky8vT9HR0dqwYUOodguwBEdQQBCcOHHCdzmmXr16KSsrS5999pl+//vf6/Dhw3K73erQoYNvfYfD4Tsaevfdd/XYY49JOnWl8+joaB07dkwdOnRQt27dJEndu3fXF198EeS9AqxFoIAg+O41qO+bPXu27rjjDqWmpmrr1q16+umnfY9FRkb63WZERNUlJRsAAACNSURBVITv12FhYXK5XIEbGDAAp/iAECkrK/N9XMOaNWtqXe+aa67R8uXLJZ26KGlZWVlQ5gNCjUABITJu3DhNmDBBmZmZatOmTa3r3Xfffdq6dasyMjKUmZmpvXv3BnFKIHS4mjkAwEgcQQEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAw0v8F34iJJDYKQjMAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "#SIBSP\n",
        "\n",
        "g = sns.catplot(x=\"SibSp\", y=\"Survived\", data=train_df, height=6, \n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "g = sns.catplot(x=\"Parch\", y=\"Survived\", data=train_df, height=6, \n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "g.despine(left=True)\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 265,
      "id": "af39a0bb",
      "metadata": {
        "id": "af39a0bb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "outputId": "ad09d532-68ac-4bb9-d81e-e6a0e07bd4fb"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 474.375x432 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "#EMBARKED\n",
        "\n",
        "g = sns.catplot(x=\"Embarked\", y=\"Survived\", hue='Sex', data=train_df, height=6, \n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "g.despine(left=True)\n",
        "g.set_xticklabels(['Cherbourg', 'Queensland', 'Southampton'])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)\n",
        "FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )\n",
        "FacetGrid.add_legend()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "VWG7fpex6zwm",
        "outputId": "8c7e50d6-70d1-432f-a2f2-a2925569ee12"
      },
      "id": "VWG7fpex6zwm",
      "execution_count": 273,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/axisgrid.py:337: UserWarning: The `size` parameter has been renamed to `height`; please update your code.\n",
            "  warnings.warn(msg, UserWarning)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<seaborn.axisgrid.FacetGrid at 0x7fe1f7eca890>"
            ]
          },
          "metadata": {},
          "execution_count": 273
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 560.775x972 with 3 Axes>"
            ],
            "image/png": "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REROV2pY+eGHHwCYOnUqoaGh9O/fH4Bly5aRnp7u/upERESkxivXaaCNGzcyfPhwAgICCAgIYNiwYWe1zhcRERFxh3KFldq1a7Ns2TLsdjsOh4Nly5ZRu7baHYuIiIj7lSusvPrqq6xevZobbriBG264gTVr1vDqq6+6uzYRERGR8jWFa9KkCTNmzHB3LeIGW347ypxvDzGyUxO6XqK7WouISNVTrpmVgwcPcvfdd9OvXz8AEhISmD59ulsLk4rx3tYkvj90gve2JhldioiIyAUpV1h57rnneOqpp/D2Lp6IiYyMZNWqVW4tTCpGXpG9xGcREZGqplxhJT8/nyuvvLLEmNlsdktBIiIiIqcrV1gJDg7m999/dzWIW7NmDaGhoW4tTERERATKucD2+eef57nnnuO3336jW7duNGnSRFcDiYiISKUoV1hp1KgRH374IXl5eTgcDgICAtxdl4iIiAhQztNAPXv25LnnnuOnn36iTp067q5JRERExKVcYWX16tVcf/31zJs3j549ezJx4kT+97//ubs2ERERkfKFFX9/f/r27cvbb7/NZ599Rk5ODiNHjnR3bSIiIiLlW7MCsGPHDlatWsVXX31F27ZtmTp1qjvrEhEREQHKGVaioqK47LLLiImJYezYsbqJoYiIiFSacoWVZcuW6QogETkv3YNKRNyp1LDy/vvv88ADD/DGG2+4GsKdbvz48W4rTESqjve2JpGQlkOe1a6wIiIVrtSwcumllwLQtm3bSilGRKom3YNKRNyp1LASFRUFQOvWrbniiisqpSARERGR05VrzcrLL79MRkYG0dHR9O3bl9atW7u7LhERERGgnGFlzpw5pKens3r1aiZMmEBubi4xMTE89NBD7q5PREREarhyNYUDCA0N5a677uLFF18kMjKS6dOnu7MuEREREaCcMysHDhxg1apVrFu3jqCgIGJiYhg3bpy7axMREREpX1h59tln6du3L7NmzcJisbi7JhERERGXMsOK3W6nSZMm3H333ZVRj4iIiEgJZa5ZMZvNHDlyBKvVWhn1iIiIiJRQrtNATZo0YejQoURFRZW4L9C9997rtsJEREREoJxhpWnTpjRt2hSn00lubq67axIRERFxKVdYeeSRR9xdh4iIiMg5lSusjBw58pw3Mvz4448rvCARERGR05UrrPzjH/9wfV1YWMi6deswm81uK0pERETklHKFlTPvutyxY0cGDRrkloJERERETleusJKZmen62uFwsHv3brKzs91WlIiIiMgp5QorAwYMcK1Z8fb2pnHjxkyePNmthYmIiJxyqm3G6e0zpOYoNazs3LmThg0bsnHjRgA+++wz1q5dS5MmTWjZsmWlFCgiIvJA3IPMXzCPYUOHG12KGKDUDrbPP/88tWrVAuDbb7/ltdde4/bbbycgIIAJEyZUSoEiIiJdbujCO29Np8sNXYwuRQxQalix2+0EBQUBsGrVKu68806io6N5/PHHSUpKqpQCRUREpGYrNaw4HA5sNp80dvgAACAASURBVBsAX3/9NZ07d3Zts9vt7q1MLorN4WTZrhRSsgoBSMkqZNmuFOwOp8GViYiI/DWlrlmJjY1lxIgRBAcH4+fnxzXXXANAUlISAQEBlVKg/HU2h5NnV/zCl/syXGNWu4NJ635ly8FjTOl3Gd5eZzf5ExER8USlzqyMHj2acePGMWDAAObPn++6IsjhcPDcc8+VefD4+Hiio6Pp1asXM2fOPO9+a9eupU2bNuzatesvli/nsmpPaomgcrov92Ww6ufUSq5IRETkwpV56XL79u3PGmvRokWZB7bb7UycOJHZs2djsVgYNGgQUVFRZ11FlJOTw8cff8xVV131F8qW0izdnVLq9iU//MGtbcMrqRoREZGLU+rMysXYuXMnzZo1IyIiAh8fH2JjY9mwYcNZ+02bNo0HHngAX19fd5VS46RmF5a6PSEth/vm/8CC7w+TkVP6viIiIkYrV1O4C5Gamkp4+J9/vVssFnbu3Flinz179pCSkkKPHj34z3/+U67jms0mgoLUFKg0jYP9ywwsu45ks+tINm9sOsB1zesT264h0VdYCK7tU0lVSnXidXINlJeXfj9FpOK5LayUxeFw8PLLL/PSSy/9pefZ7U4yM/PcVFX1EBsZxve/Z553e4v6tUk6nofDCU4nfHPwGN8cPMYLK36mc7NgekeG0r1lCHV8DPvxkCrGcfIqM4dDv58iniQ0NNDoEiqE296NLBYLKSl/rp1ITU3FYrG4Hufm5vLrr79y1113AZCens7o0aOZMWMG7dq1c1dZNULsFRa2HDx2zkW2N7VqwEv9LuN4fhEb9qazbm86O//IAsDucLL14DG2HjyGr7cXXVrUJzoylBta1Mevlu6yLSIixjA5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vPrqq+4qR0RERKoot4WVnTt30qxZMyIiIvDx8SE2NpYNGzaU2Gfjxo3cfvvtAERHR/P111/jdDrdVZKIiIhUQW4LK6mpqYSHh7seWywWUlNTz9qnYcOGAHh7exMYGMjx48fdVZKIiIhUQd5GF/BXmc0mgoJqG12GiJwm0L8WHM8n0L+Wfj9FpMK5LaxYLBZSUlJcj1NTU7FYLGftc+TIEcLDw7HZbGRnZxMcHFzqce12J5mZeW6pWUQuTNx1Ecz93yFGXNNEv58iHiQ0NNDoEiqE204DtWvXjsTERJKTk7FaraxcuZKoqKgS+0RFRfHZZ58BsHbtWjp37ozJZHJXSSLiJl0vCeHdwVfR9ZIQo0sRkWrI5HTjitbNmzczZcoU7HY7AwcOZPTo0UybNo22bdvSs2dPCgsL+fvf/84vv/xCvXr1eOONN4iIiCj1mEVFdv3lJiIiUg7VZWbFrWHFHRRWREREyqe6hBV1sBURERGPprAiIiIiHk1hRURERDyawoqIiIh4NIUVERER8WgKKyIiIuLRFFZERETEoymsiIiIiEerck3hREREpGbRzIqIiIh4NIUVERER8WgKKyIiIuLRFFZERETEoymsiIiIiEdTWBERERGPprAiIiIiHk1hRURERDyawoqIiIh4NIUVERER8WgKKyJV2GWXXUb//v1dHzNnziz3c7dv386oUaMu6vVHjhzJrl27Lui548aNY82aNRf1+gD3338/11xzzUV/LyLiubyNLkBELpyfnx9Lly415LXtdrshr3umuLg48vPzWbRokdGliIibKKyIVENRUVHExsYSHx+P2Wxm0qRJvP766yQlJXH//fczdOhQAHJycnjwwQdJSkriuuuu44UXXsDLy4vnn3+eXbt2UVhYSHR0NI899pjruDExMWzbto24uDjX6zkcDp599lksFguPPfYYr776Kjt27MBqtTJ8+HCGDBmC0+lk0qRJbN26lYYNG1KrVq0K+V6vv/56tm/fXiHHEhHPpLAiUoUVFBTQv39/1+NRo0bRt29fABo2bMjSpUuZMmUK48aNY8GCBVitVvr16+cKKzt37mTVqlU0atSIuLg41q1bR58+fXjiiScICgrCbrdzzz33kJCQQGRkJABBQUF89tlnACxcuBC73c7TTz9Nq1atGD16NIsWLSIwMJBPP/0Uq9XKkCFD6NKlC7/88gsHDx5k1apVZGRkEBsby8CBA8/6nmbNmsXy5cvPGu/UqRPjx4+v8H9DEfF8CisiVVhpp4F69uwJQOvWrcnLyyMgIAAAHx8fsrKyALjyyiuJiIgAIDY2lu+++44+ffqwevVqFi9ejM1mIz09nQMHDrjCyqkwdMqECROIiYlh9OjRAGzdupW9e/eydu1aALKzs0lKSuLbb78lNjYWs9mMxWKhc+fO56w7Li6uxKyNiIjCikg1deo0i5eXFz4+Pq5xLy8vbDYbACaTqcRzTCYTycnJfPDBB3zyySfUq1ePcePGUVhY6NrH39+/xHM6dOjA9u3bue+++/D19cXpdDJ+/Hi6detWYr/NmzeXq27NrIjImXQ1kEgNtnPnTpKTk3E4HKxevZqOHTuSm5uLv78/gYGBZGRkEB8fX+oxBg0aRPfu3RkzZgw2m42uXbuyYMECioqKADh48CB5eXl06tSJ1atXY7fbSUtLO+86k7i4OJYuXXrWh4KKSM2lmRWRKuzMNSvdunXj6aefLvfz27Vrx6RJk1wLbHv16oWXlxeXX345MTExhIeHc/XVV5d5nHvvvZfs7GzGjh3Lq6++yuHDhxkwYABOp5Pg4GCmT59Or169+Oabb+jbty+NGjWiffv2F/Q9n2nYsGH89ttv5OXlceONNzJ58uSzZnVEpGozOZ1Op9FFiIiIiJyPTgOJiIiIR1NYEREREY+msCIiIiIeTWFFREREPFqVuxrIarVx4kS+0WWIiIh4vNDQQKNLqBBVbmblzCZWIiIiUr1VubAiIiIiNYvCioiIiHg0hRURERHxaAorIiIi4tEUVkRERMSjKayIiIiIR1NYEREREY9W5ZrCSfnYbDZWr1nN8pXLSEtNI8wSxi2xt9I3pi9ms9no8qSa0M+ZiFQGk9PpdLrjwM888wybNm0iJCSEFStWnLXd6XQyefJkNm/ejJ+fHy+//DJXXHFFmcctKrKTmZnnjpKrDZvNxnPPj2fT5k1nbevRvQeTXvwn3t7KqXJx9HMm4vnUwbYMAwYMYNasWefdHh8fT2JiIuvWrWPSpEm88MIL7iqlxlm9ZvU530AANm3exJq1ayq3IKmW9HMmIpXFbX/2dOrUiUOHDp13+4YNG7jtttswmUy0b9+erKws0tLSCAsLc1dJNcbylctK3f7yv17inRlvV1I1Ul1lZ2eXun35imX0i+1XSdWISHVm2Bxtamoq4eHhrsfh4eGkpqaWGVbMZhNBQbXdXV6VlpGeXup2u91OZmZmJVUjNVVaWqp+V0WkQlS5E8p2u1NrVsrQIDSUIykp593u7+9Pi+YtKrEiqY4OJh4kP//8d0A/npnJ3HmL6d2rNz4+PpVYmYicUl3WrBgWViwWCymnvaGmpKRgsViMKqdauSX2Vnbt2nXe7U8+/pSm5+WiLV+xnCkvTz7v9sLCQia/9E/ee/897rxjMLf1v52AgIBKrFBEqgvD+qxERUXx+eef43Q6+fHHHwkMDNR6lQrSN6YvPbr3OOe2Ht17ENMnpnILkmqptJ+zBg0aYDKZAMjISOedGe/Qf8CtvD39LdLS0yqxShGpDtx26fKTTz7Jjh07OH78OCEhITz66KPYbDYAhg4ditPpZOLEiXz11Vf4+/szZcoU2rVrV+Zxdely+dhsNtasXcPyFctITU3FYrFwS79biekTo/4XUmFK+zk7fPgwCxYtYNXqlVitVtdzvL29ie7dh+FDh9OihU5HirhTdTkN5Law4i4KKyJVy7FjR1nyyRI+/e+nZOeUvIKoyw1dGDF8JFddeZVrJkZEKo7CikEUVkSqpry8PJavWM7CRQtISS25ALztFW0ZPmw43breqJk/kQqksGIQhRWRqs1ms7Fh4xfMnTeX/Qf2l9gW0SSCYUOHE9MnBl9fX4MqFKk+FFYMorAiUj04nU52fLuDufPm8L/v/ldiW/369blj0GAG3DaAunXrGlShSNWnsGIQhRWR6ichIYF5C+ay8cuNOBwO17i/vz+33tKfIYOHlGgiKSLlo7BiEIUVkerr8OHDLFy8gOUrllNYWOgaN5vN9Lq5F8OHDqdly1YGVihStSisGERhRaT6y8zM5NP/fsKST5dw4sSJEtuuu7YzI4aPoOPVHXUFkUgZFFYMorAiUnMUFBSwctUK5i+Yzx9H/iixLbJNJMOHDadH95vw9q5ydw4RqRQKKwZRWBGpeWw2G5s2b2Le/Lkk7E0osa1Rw0YMHTKMfrH98PPzM6hCEc+ksGIQhRWRmsvpdPL9D98zd94cvtn+TYlt9erVY9CAQQwaeAdBQUEGVSjiWRRWDKKwIiIA+/btY/7Ceaz/Yj12u9017uvrS7/YWxh651AaN25sYIUixlNYMYjCioicLiUlhUVLFrJ02VLy8/Nd415eXkTdFMXwoSOIjIw0sEIR4yisGERhRUTOJSsri/9+/l8WL1nE8ePHS2zr2PEaRgwbwXXXXqcriKRGUVgxiMKKiJSmsLCQNWtXM2/+PJIPJZfY1vLSlgwfNpybe/bSFURSIyisGERhRUTKw26389WWr5g3fy679+wusc0SZmHIkKHc2u9WateubVCFIu6nsGIQhRUR+SucTic7d/7E3Plz2bJ1S4ltgYF1GXDb7Qy+YzD164cYVKGI+yisGERhRUQu1MGDB5m3YB5r163BZrO5xn18fIjp05dhQ4bRtGlTAysUqVgKKwZRWBGRi5WWnsaSJYv5bOln5ObmusZNJhPdb+zO8KEjaNu2rYEVilQMhRWDKKyISEXJycnh82Wfs2jxIjIy0ktsa39Ve4YPG8EN19+Al5eXQRWKXByFFYMorIhIRSsqKmLd+rXMmz+Pg4kHS2xr0bwFpOt+TwAAIABJREFUw4cNp3evaGrVqmVQhSIXRmHFIAorIuIuDoeDbV9vY978ufz4048ltjVoEMqdg+/ktltvIyAgwKAKRf4ahRWDKKyISGXYvXs38xbMZXP8Zk7/b7JOnTrc3v92Bg++k9AGoQZWKFI2hRWDKKyISGX6/fffmb9wPqvXrMJqtbrGvb29ie7dh+FDh9OiRQsDKxQ5P4UVgyisiIgRjh07ypJPlvDpfz8lOye7xLauXboyYtgIrrzyKrXzF4+isGIQhRURMVJuXi7LVyxn4cIFpKalltjW9oq2DB82gm5du2E2mw2qUORPCisGUVgREU9gs9n4YsN65s2fx/4D+0tsaxrRlGFDh9EnOgZfX1+DKhRRWDGMwoqIeBKn08n2HduZO38u3333vxLb6tevzx2DBjPgtgHUrVvXoAqlJlNYMYjCioh4qoSEBOYtmMvGLzfi+H/27jy+qTJR4/iTpvteKqSFFlxARakK6MhcUbHIlFIYlOKwyUUdQFFUEFnGAWRAGBzZOqIgemVUFEVcWFoURQUFRlzAouICUihIy9KFlm4kzf2jEKlAG0rTk6S/7+fTT5qck5MHRPpw3ve8p7LS8XpQUJB6/7m3+t3RXzExMQYmRGNDWTEIZQWAu9u/f7+WvrFUq9NXqby83PG62WxWt1u7adCAQWrduo2BCdFYUFYMQlkB4CkKCgr01tvL9eZbb6qwsLDatk7Xd9KdgwarQ/sOXEEEl6GsGISyAsDTlJWVaXX6ai19/TX9euDXatsuv+xyDRp4p7rc3EW+vr4GJYS3oqwYhLICwFNZrVZ9sv5jvfraq/rhxx+qbWvRvIUG9B+glB49FRgYaFBCeBvKikEoKwA8nd1u11dff6Ulry7R51v+W21bZGSk+vbpq9Q+fRUZGWlQQngLyopBKCsAvMnPP/+s115/VR98+IFsNpvj9YCAAPXq2UsD+g1U8+bNDUwIT0ZZMQhlBYA3OpBzQG8se0MrV61QaWmp43UfHx8l3pKoQQPu1OWXX25gQmNt3LRRr762RIMG3qkb/ucGo+N4DMqKQSgrALxZ4dFCvfPuO1r25hvKz8+vtu3ajtdq0MA7df0frm90VxDddc8Q/fjTj7rs0sv0nxdfMjqOx6CsGISyAqAxKC8v15r31ui1pa8qe192tW1tWrfRoIGD1DXx1kZzBdFf+t+h7H3Zio+L17LX3zQ6jsegrBiEsgKgMbHZbPr0s0+15NVX9N3331XbFmOJUf9+A9SrZy8FBwcblLBhUFbqhrJiEMoKgMbIbrfrm8xvtOTVV7Rx08Zq28LCwpV6ex/d0fcONWkSbVBC16Ks1A1lxSCUFQCN3e7du/Xq0lf1/tr3ZLVaHa/7+/urR3KKBvYfoPj4lgYmrH+UlbqhrBiEsgIAVQ4eOqhlb76hd959RyUlv/29aDKZdPNNN+vOgXfqyivbGZiw/lBW6oayYhDKCgBUV1xcrHdXvqs33nhdh48crrbtmmvaa9CAQfqfP/6PfHx8DEp4/igrdUNZMQhlBQDOrKKiQms/WKtXly5RVlZWtW0XXXiRBg0cpD91S5Kfn58xAc8DZaVuKCsGoawAQM0qKyu1afMmvfraEm37Zlu1bU2bNlW/O/rrtt63KSQkxKCE546yUjfeUlZcek5ww4YNSkpKUrdu3bRo0aLTtv/6668aPHiwbrvtNvXq1Uvr1693ZRwAaBR8fHzU+YbOWvDMQi1a+LxuvulmxyJyhw4d0vxnn1bvPn/WM8/O16HDhwxOC9TOZWdWbDabkpKStHjxYlksFvXt21dz5sxR69atHftMmjRJbdu21cCBA7Vz504NHz5cH330UY3H5cwKAJy7vXv36rXXX9Oa9zJUUVHheN3X11fdk5I1aMBAXXjhRQYmrBlnVuqGMyu1yMzMVKtWrRQfHy9/f3+lpKRo3bp11fYxmUwqLi6WJBUVFalZs2auigMAjVrLli01YdwEvf3mOxoyeIjCQqt+iFmtVq1OX6UBdw7Q2PGP6ptvtsnDZgegEXDZOs25ubmKiYlxPLdYLMrMzKy2z8iRI/XXv/5VS5YsUWlpqRYvXlzrcc1mkyIjvXulRgBwlcjIYE0YP0YPjrxPy99+Wy+99IoO5ORIkj7b+Jk+2/iZrrn6at1z913qmniL21xB5GP2cTzyM6DxMfSmEunp6br99tt1zz33aOvWrRo3bpxWr15d4/8cNpudYSAAOG8m9e6VqpTk3vpw3Qda8toS7dq1S5K07Ztv9NCo0WoZ31IDBwxU96RkBQQEGJq20lbpeORngPMYBqqFxWJRzom2LlWdabFYLNX2Wb58uZKTkyVJ7du3V3l5+Wl3GQUAuM7JOSuv/GeJ5s6ep44dr3Vs25u9VzP/NVN97rhdL73yko4ePWpgUjRmLisrCQkJysrKUnZ2tioqKpSenq7ExMRq+8TGxmrz5s2SpF27dqm8vFxNmjRxVSQAwFmYTCZ1ur6T5qfN14svLFbXxK6Os9x5eXla+NwC3Z56m9KenlftH6JAQ3DpOivr16/XjBkzZLPZlJqaqhEjRigtLU3t2rVT165dtXPnTk2cOFElJSUymUwaO3asOnfuXOMxuRoIABrG/v37tfSNpVqdvkrl5eWO181ms7rd+icNGjhIrS9pXcMR6g9XA9WNtwwDsSgcAKBG+fn5euvt5Vr+9nIVFhZW29bp+k66c9BgdWjfwbGWiytQVuqGsmIQygoAGKOsrEyr01dr6euv6dcDv1bb1vbytho08E51ubmLzGZzvX82ZaVuKCsGoawAgLGsVqs+Wf+xlry6RD/+9GO1bS2at9CA/gOU0qOnAgMD6+0zKSt1Q1kxCGUFANyD3W7XV19/pSWvLtHnW/5bbVtkZKT6pt6hvn36KiIi4rw/i7JSN5QVg1BWAMD9/Pzzz3p16RJ9uO5D2Ww2x+uBgYHqmdJTA/oNVPPmzet8fMpK3VBWDEJZAQD3dSDngN5Y9oZWrlqh0tJSx+s+Pj7qektXDRo4SJdddvk5H5eyUjeUFYNQVgDA/RUeLdQ777ytZcuXnbbY57Udr9WdgwbrD9f9wekriCgrdUNZMQhlBQA8R3l5uda8t0avLX1V2fuyq21r07qNBg28U10Tu8rXt+a7v1BW6oayYhDKCgB4HpvNpk8/26BXlryi73d8X21bjCVG/fsNUK+evRQcfOabFFJW6oayYhDKCgB4Lrvdrm3fbNOrry3Rxk0bq20LCwtXap9U3dH3DjWJqn7rFcpK3XhLWTFPmTJlitEhzkVlpV1lZceNjgEAqAOTyaTYmFj9qVuSErskqry8TLt371ZlZaUqKsq17ZttWv7Wmzp48KBatWypkJAQpWek64MP18pms6msrEwXRF+g1pe0dty7CGcXEmLs3bLrC2dWAACGOnjooJa9+YbeefcdlZT89ve7yWRSdHS0Dh8+fNp7utzcRdP+8UStc10aO285s0JZAQC4heLiYr274h29sewNHT5yekH5vb//baJ6pvRsgGSei7JiEMoKAHi3iooKvb/2fc2eO6va3Z5/76qEq/TcgkUNmMzzeEtZYcAPAOBW/P391atnL0WE17xMf25ubgMlgtEoKwAAt2SJsdS83VLzdngPygoAwC31Svlzzdt71rwd3oOyAgBwSz2Se6jLzV3OuK3LzV2U3D25YQPBMEywBQC4LavVqvfef09Pzf6XKioq5O/vr7Fjxim5e7LMZrPR8dweE2wBAHAxX19f9UzpKUuzqvkplmYW9UzpSVFpZCgrAADArVFWAACAW6OsAAAAt0ZZAQAAbo2yAgAA3BplBQAAuDXKCgAAcGuUFQAA4NYoKwAAwK1RVgAAgFujrAAAALdGWQEAAG6NsgIAANwaZQUAALg1ygoAAHBrlBUAAODWKCsAAMCtUVYAAIBbo6wAAAC3RlkBAABujbICAADcGmUFAAC4NcoKAABwa5QVAADg1lxaVjZs2KCkpCR169ZNixYtOuM+GRkZ6tGjh1JSUjRmzBhXxgEAAB7I11UHttlsmjp1qhYvXiyLxaK+ffsqMTFRrVu3duyTlZWlRYsWaenSpYqIiNCRI0dcFQcAAHgol51ZyczMVKtWrRQfHy9/f3+lpKRo3bp11fZZtmyZBg0apIiICElSdHS0q+IAAAAP5bIzK7m5uYqJiXE8t1gsyszMrLZPVlaWJKl///6qrKzUyJEjddNNN9V4XLPZpMjI4HrPCwBwXz5mH8cjPwMaH5eVFWfYbDbt2bNHr7zyinJycnTnnXdq1apVCg8Pr+E9dhUUlDRgSgCA0SptlY5HfgY4r2nTMKMj1AuXDQNZLBbl5OQ4nufm5spisZy2T2Jiovz8/BQfH68LL7zQcbYFAABAcmFZSUhIUFZWlrKzs1VRUaH09HQlJiZW2+fWW2/Vli1bJEl5eXnKyspSfHy8qyIBAAAP5LJhIF9fX02ePFlDhw6VzWZTamqq2rRpo7S0NLVr105du3bVjTfeqI0bN6pHjx4ym80aN26coqKiXBUJAAB4IJPdbrcbHeJcHD9uY7wSABqZv/S/Q9n7shUfF69lr79pdByPwZwVAAAaSHBwcLVHNC6UFQCA2xs2dLg6tO+gYUOHGx0FBqhxGKh9+/YymUxnffPXX3/tklA1YRgIAADneMswUI0TbLdu3SpJmjdvnpo2barevXtLklauXKlDhw65Ph0AAGj0nBoG+uijjzRo0CCFhoYqNDRUAwcOPG3pfAAAAFdwqqwEBwdr5cqVstlsqqys1MqVK5nkBAAAGoRTly7v27dP06dP19dffy2TyaQOHTroscceU1xcXENkrIY5KwAAOMdb5qywzgoAAF7KW8qKU8NAu3fv1pAhQ9SzZ09J0g8//KBnn33WpcEAAAAkJ8vKpEmTNGbMGPn6Vl08dPnllysjI8OlwQAAACQny0ppaamuuuqqaq+ZzWaXBAIAADiVU2UlKipKe/fudSwQ995776lp06YuDQYAACA5OcE2OztbkyZN0tatWxUeHq64uDjNmjVLLVq0aIiM1TDBFgAA53jLBFunyorNZpPZbFZJSYkqKysVGhraENnOiLICAIBzvKWsODUM1LVrV02aNEnffPONQkJCXJ0JAADAwakzK6Wlpfr444+VkZGh77//Xl26dFGPHj107bXXNkTGajizAgCAc7zlzMo5LwpXWFio6dOna9WqVdqxY4ercp0VZQUAAOd4S1lxahhIkrZs2aIpU6aoT58+Ki8v17x581yZCwAAGMRqterN5W/rL/0H6cYuXfWX/oP05vK3ZbPZzvvYGzZsUFJSkrp166ZFixY59R6nzqwkJiaqbdu2Sk5OVmJioqE3MeTMCgAAzqnLmRWr1aqHR4/R+2s/PG1b0p9uVdrc2Y5FYs+VzWZTUlKSFi9eLIvFor59+2rOnDlq3bp1je9z6tNWrlxp6BVAAACgYbzz7sozFhVJen/th3p3xSr1Tb29TsfOzMxUq1atFB8fL0lKSUnRunXrzq+sPP/88xo2bJjmzp3rWBDuVBMnTqxTWAAA4J7eXP5WrdvrWlZyc3MVExPjeG6xWJSZmVnr+2osK5dccokkqV27dnUKBQAAPMuBnJwat/964EADJflNjWUlMTFRknTppZfqyiuvbJBAAADAOLExMTpw4OyFpXlsbJ2PbbFYlHNKGcrNzZXFYqn1fU5dDTRz5kwlJydr3rx5+umnn+ocEgAAuLc7+qae1/aaJCQkKCsrS9nZ2aqoqFB6errjxEhNnF5n5dChQ1qzZo0yMjJ07NgxJScn6/77769z4LriaiAAAJxTl6uBbDabHhr1yFmvBvr3vDkym811zrR+/XrNmDFDNptNqampGjFiRK3vOedF4X788Ue98MILWrNmjb799ts6h60rygoAAM6p66JwVqtV765YpTeXv6VfDxxQ89hY3dE3Vbff9ufzKip15VRZ2bVrlzIyMrR27VpFRkYqOTlZSUlJio6OboiM1VBWAABwjresYOtUWenXr5969Oih7t27OzURxpUoKwAAOMdbykqti8LZbDbFxcVpyJAhDZEHAACgmlqvBjKbzTpw4IAqKioaIg8AAEA1Ti23HxcXpwEDBpx2X6C7777bZcEAAAAkJ8tKy5Yt1bJlS9ntdh07dszVmQAAABzO+dJlozHBFgAA59T50mVbpd7+er/e+DJbBwpKFRsZpH7Xxiu1Y5zMPqffK9BZf/vb3/TJJ58oOjpaq1evdvp9TpWVwYMHn/FGhi+//PK5pawHlBUAAJxTl7JitVVq5Gtb9d53py+53/3KGM0f2F6+ZqcWwD/NF198oeDgYI0fP/6cyopTw0Djx493fF9eXq61a9casigMAABwrbe/3n/GoiJJ732Xo7e37tdfro2v07Gvu+467du375zf51RZ+f1dlzt27Ki+ffue84cBAAD39saX2TVuX/ZFdp3LSl05VVYKCgoc31dWVurbb79VUVGRy0IBAABjHCgorXH7r7VsdwWnykqfPn0cc1Z8fX3VokULTZ8+3aXBAABAw4uNDNKvhWVn3d48MqgB01SpsaxkZmYqNjZWH330kSTpnXfe0fvvv6+4uDi1bt26QQICAICG0+/aeH21J/+s2/9yXcMOAUm1rGD7+OOPy8/PT1LVDN7Zs2fr9ttvV2hoqCZPntwgAQEAQMNJ7Rin7lfGnHFb9ytjlNohrs7HfuSRR9S/f3/t3r1bN910k958802n3lfjmRWbzabIyEhJUkZGhvr166ekpCQlJSWpd+/edQ4LAADck9nHpPkD2+vtrfu17Its/VpQquaRQfrLdfFK7XB+66zMmTOnTu+rsaxUVlbKarXK19dXmzdv1rRp0xzbbDZbnT4QAAC4N1+zj/5ybXyDX/VzNjWWlZSUFN15552KiopSYGCgrr32WknSnj17FBoa2iABAQBA41brCrbbtm3ToUOHdMMNNzhuYrh7926VlJToyiuvrPHgGzZs0PTp01VZWak77rhDw4cPP+N+77//vh566CEtX75cCQkJNR6TFWwBAHBOXZfbdze1Xrp8zTXXnPbaRRddVOuBbTabpk6dqsWLF8tisahv375KTEw87Sqi4uJivfzyy7r66qvPITYAAGgs6ra4vxMyMzPVqlUrxcfHy9/fXykpKVq3bt1p+6WlpWnYsGEKCAhwVRQAAODBnFoUri5yc3MVE/PbpU8Wi0WZmZnV9vnuu++Uk5OjLl266P/+7/+cOq7ZbFJkZHC9ZgUAAO7LZWWlNpWVlZo5c6b++c9/ntP7bDY7c1YAAHCCt8xZcdkwkMViUU7Ob3dtzM3NlcVicTw/duyYfvrpJ/3v//6vEhMTtW3bNo0YMULbt293VSQAAOCBXHZmJSEhQVlZWcrOzpbFYlF6erpmz57t2B4WFqbPP//c8Xzw4MEaN25crVcDAQCAxsVlZcXX11eTJ0/W0KFDZbPZlJqaqjZt2igtLU3t2rVT165dXfXRAADAi9S6zoq7YZ0VAACcw5wVAACABkBZAQAAbo2yAgAA3BplBQAAuDXKCgAAcGuUFQAA4NYoKwAAwK1RVgAAgFujrAAAALdGWQEAAG6NsgIAANwaZQUAALg1ygoAAHBrlBUAAODWKCsAAMCtUVYAAIBbo6wAAAC3RlkBAABujbICAADcGmUFAAC4NcoKAABwa5QVAADg1igrAADArVFWAACAW6OsAAAAt0ZZAQAAbo2yAgAA3BplBQAAuDXKCgAAcGuUFQAA4NYoKwAAwK1RVgAAgFujrAAAALdGWQEAAG6NsgIAANwaZQUAALg1ygoAAHBrlBUAAODWKCsAAMCtUVYAAIBbo6wAAAC3RlkBAABujbICAADcmkvLyoYNG5SUlKRu3bpp0aJFp21fvHixevTooV69emnIkCHav3+/K+MAAAAP5LKyYrPZNHXqVL3wwgtKT0/X6tWrtXPnzmr7tG3bVm+99ZZWrVqlpKQkPfXUU66KAwAAPJTLykpmZqZatWql+Ph4+fv7KyUlRevWrau2T6dOnRQUFCRJuuaaa5STk+OqOAAAwEP5uurAubm5iomJcTy3WCzKzMw86/7Lly/XTTfdVOtxzWaTIiOD6yUjAABwfy4rK+dixYoV+vbbb7VkyZJa97XZ7CooKGmAVAAAeLamTcOMjlAvXFZWLBZLtWGd3NxcWSyW0/bbtGmTFi5cqCVLlsjf399VcQAAgIdy2ZyVhIQEZWVlKTs7WxUVFUpPT1diYmK1fb7//ntNnjxZCxYsUHR0tKuiAAAAD2ay2+12Vx18/fr1mjFjhmw2m1JTUzVixAilpaWpXbt26tq1q+666y799NNPatq0qSQpNjZWCxcurPGYx4/bGAYCgEbms1+O6JUv9mnwdXHqfDH/uHWWtwwDubSsuAJlBQAan8GvfK0fDhbr8mahemVwB6PjeAxvKSusYAsAcHslx23VHtG4UFYAAIBbo6wAAAC3RlkBAABujbICAADcGmUFAAC4NcoKAABwa5QVAADg1igrAADArVFWAACAW6OsAAAAt0ZZAQAAbo2yAgAA3BplBQAAuDXKCgAAcGuUFQAA4NYoKwAAwK1RVgAAgFujrAAAALdGWQEAAG6NsgIAANwaZQUAALg1ygoAwG1ZK+1auT1HOUfLJUk5R8u1cnuObJV2g5OhIZnsdrtH/Rc/ftymgoISo2MAAFzMWmnXY6t36OOfD5+27ZY2F2hGz7by9TEZkMxzNG0aZnSEesGZFQCAW1qReeCMRUWSPv75sDK+z23gRDCKr9EBAACNk7XSrtyiMv1a+NvX/lMe80qO1/j+ldtz9Od2MQ2UFkairAAAXMJutyuv5Hi1EvJrYZn2H616zD1aJtt5TETIKSqvv7Bwa5QVAECdFZdbfzszcvS3syL7C8t0oLBMZdbKcz5miL9ZzSMClVtUrqNl1rPuFxMWcD7R4UEoKwCAszpuq9SBo+X6tbD0RBGp+v7kmZLCGsrE2fj6mBQbHqDmEYFqERGk5hGBjq8W4YGKCPKVyWTSyu05mrb2p7Me588JDAE1FpQVAGjEKu12HS6uqD5f5OhvQzYHi8pVl5GapqH+anGyhISfUkYiAtU0NEBmJ67iSbnSos925531aqCUKyx1SAZPxKXLAODljpYdrz5n5JTHnKNlqqjDxJHwQN9qJeRkEWkeEajY8EAF+NbPxabWSrsyvs/Vkx/uVIWtUv5mH42/tbVSrrA4VXgaO2+5dJkzKwDg4cqO204M1ZycL1JabR5JcbntnI8Z4OtzxqGaFicKSlhgw/z48PUx6c/tYvTSlmztzS9VTHgAVwA1QpQVAHBztkq7DhafKCMF1Ydpfi0s0+FjFed8TB+T1Cw0QC0iTx+maRERqCYh/vIxceYC7oGyAgAGs9vtKig9Xm145tQhm5yi8jotLx8V5FdtmObUMmIJC5CfmXVB4RkoKwDQAEoqbL8VkaNl2l9QWu1y39Lj536Jb5CfT7UJrC0ig9Q8/Le5I8H+Zhf8SoCGR1kBgHpgtVUqp6jcscbIr7/7yi+teTXWMzH7mBQTFlBt8uqpj5FBfjIxVINGgLICAE6w2+06cqzi9DJytGoeycHictXlRsAXhPifts5Ii8iq75uGBnCjPkCUFQBwKCqznrbOyMkraw4cLVd5HVZjDQ0w/24Ca9Apl/gGKNCPoRqgNpQVAI1GubVSB46evt7IyTMkNS3tfjZ+ZpNiw08fpjn5fXignwt+JUDjQlkB4DVslXYdKi4/rYScLCaHis/9El+TTqzGGhlUbZ2Rk6XkglAu8QVcjbICwGPY7XYVllpPW2fk1KEaax0mjkQE+p5xEmvziCDFhAXIv55WYwVQN5QVAG6l9Ljtd0Wk+hmSYxV1W4311DVGfn+/mtAA/ioE3Bn/h3opa6VdGd/lasW3OcotKpclLEC928Uo5Urup4H6U5c/Z9ZKu3KLqq6gObWEnCwmeSV1uMTXJFnCAtQ8MsgxTHPqGZImwVziC3gyl5aVDRs2aPr06aqsrNQdd9yh4cOHV9teUVGhcePG6bvvvlNkZKTmzp2ruLg4V0ZqFKyVdj22eke1O5XmFpUr89ej+mx3nmb0bMvlkDhvNf05W/fzYd31h3jlFpWfNlSTW1SuOtw3T02C/U6bvHrye0tYIH+mAS/msrJis9k0depULV68WBaLRX379lViYqJat27t2OfNN99UeHi4PvjgA6Wnp2vWrFmaN2+eqyI1Ghnf5Z7xluqS9PHPh/Xg8kxd2CS4gVPB22TllejL7MIzbtu0O0+bdued0/FC/M2O4Znf36+meUSggrjEF2i0XFZWMjMz1apVK8XHx0uSUlJStG7dumpl5aOPPtLIkSMlSUlJSZo6darsdjuna8/Tim9zatz+ZXbhWX/IAK7i62NSbHhAtTv4nnqGJCLQl//3AZyRy8pKbm6uYmJ+u423xWJRZmbmafvExsZWBfH1VVhYmPLz89WkSZOzHtdsNikykrMCNTlUhzuwAvUtNMCsSSlXKC4qSPFRQWoWFsh8KdRZWJCflF+qsCA/fgY0Qh43wdZms6ugoMToGG6taYi/DhSWnXV722ah+lfvKxowEbzR2BXf64eDxWfdfkl0iBIviqp6Yrer6GhpAyWDNxp6fbyWfLlPd14bx8+Ac9C0aZjREeqFy8qKxWJRTs5vwxG5ubmyWCyn7XPgwAHFxMTIarWqqKhIUVFRrorUaPRuF6PMX4+edXvf9s0VEx7YgIngje64prmmrf3prNv/nBBz1m3Auep8cbQ6XxxtdAwYxGUrHSUkJCgrK0vZ2dmqqKhQenq6EhMTq+2TmJiod955R5L0/vvvq1OnToxZ14OUKy26pc0FZ9x2S5sLlHKF5YzbgHPBnzMADcVkt9vrcBGhc9avX68ZM2bIZrMpNTVVI0aMUFpamtq1a6euXbuqvLxcY8eO1Y4dOxQREaHd3VJNAAAgAElEQVS5c+c6JuSezfHjNk4BOsFaaVfG97lauT1HOUXligkL0J8TYpRyBeusoP7w5wxwb94yDOTSsuIKlBUAAJzjLWWFG14AAAC3RlkBAABujbICAADcGmUFAAC4NcoKAABwa5QVAADg1igrAADArXncOisAAKBx4cwKAABwa5QVAADg1igrAADArVFWAACAW6OsAAAAt0ZZAQAAbo2yAgAA3BplBQAAuDXKCgAAcGuUFQAA4NYoK4AHa9u2rXr37u34WrRokdPv/fzzz3Xvvfee1+cPHjxY27dvr9N7J0yYoPfee++8Pn/Hjh3q16+fUlJS1KtXL2VkZJzX8QC4J1+jAwCou8DAQK1YscKQz7bZbIZ87qkCAwP15JNP6sILL1Rubq5SU1PVuXNnhYeHGx0NQD2irABeKDExUSkpKdqwYYPMZrOmTZumOXPmaM+ePfrrX/+qAQMGSJKKi4s1fPhw7dmzR9dff72mTJkiHx8fPf7449q+fbvKy8uVlJSkhx56yHHc5ORkbdq0SUOHDnV8XmVlpR577DFZLBY99NBDmjVrlrZs2aKKigoNGjRI/fv3l91u17Rp07Rx40bFxsbKz8/vvH+dF110keN7i8WiJk2aKC8vj7ICeBnKCuDBysrK1Lt3b8fze++9Vz169JAkxcbGasWKFZoxY4YmTJigpUuXqqKiQj179nSUlczMTGVkZKh58+YaOnSo1q5dq+7du2v06NGKjIyUzWbTXXfdpR9++EGXX365JCkyMlLvvPOOJOn111+XzWbTo48+qjZt2mjEiBF64403FBYWprfeeksVFRXq37+/brjhBu3YsUO7d+9WRkaGDh8+rJSUFKWmpp72a3rhhRe0atWq016/7rrrNHHixLP+XmRmZur48eNq2bJl3X9DAbglygrgwWoaBuratask6dJLL1VJSYlCQ0MlSf7+/jp69Kgk6aqrrlJ8fLwkKSUlRV999ZW6d++uNWvWaNmyZbJarTp06JB27drlKCsny9BJkydPVnJyskaMGCFJ2rhxo3788Ue9//77kqSioiLt2bNHX3zxhVJSUmQ2m2WxWNSpU6cz5h46dGi1szbOOHjwoMaOHasnn3xSPj5MxQO8DWUF8FInh1l8fHzk7+/veN3Hx0dWq1WSZDKZqr3HZDIpOztbL774opYvX66IiAhNmDBB5eXljn2CgoKqvad9+/b6/PPPdc899yggIEB2u10TJ07UjTfeWG2/9evXO5X7XM+sFBcX695779Xo0aN1zTXXOPUZADwL/wQBGrHMzExlZ2ersrJSa9asUceOHXXs2DEFBQUpLCxMhw8f1oYNG2o8Rt++fXXzzTfr4YcfltVqVefOnbV06VIdP35ckrR7926VlJTouuuu05o1a2Sz2XTw4EF9/vnnZzze0KFDtWLFitO+zlRUKioq9MADD6h3797q3r37+f+GAHBLnFkBPNjv56zceOONevTRR51+f0JCgqZNm+aYYNutWzf5+PjoiiuuUHJysmJiYtShQ4daj3P33XerqKhI48aN06xZs7R//3716dNHdrtdUVFRevbZZ9WtWzf997//VY8ePdS8efN6OQuyZs0affnllyooKHDMo5k5c6batm173scG4D5MdrvdbnQIAACAs2EYCAAAuDXKCgAAcGuUFQAA4NYoKwAAwK1RVgAAgFvzuEuXKyqsKiwsNToGAABur2nTMKMj1AuPO7Py+xU3AQCAd/O4sgIAABoXygoAAHBrlBUAAODWKCsAAMCtUVYAAIBbo6wAAAC3RlkBAABuzWVl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goB58+YhPz8fq1evRmpqKrKystDd3Y3f/OY3KCkpQVhYGN544w3ExcX1e0ybTeAyEBER0Q3wlWUgj4YVT2BYISIiujG+ElZ4BVsiIiKSNYYVIiIikjWGFSIiIpI1hhUiIiKSNYYVIiIikjWGFSIiIpI1hhUiIiKSNa+7zgoRERENL+ysEBERkawxrBAREZGsMawQERGRrDGsEBERkawxrBAREZGsMawQERGRrDGsEBERkayppS6APGfp0qXYs2cPoqKisH37dqnLIR9UW1uL3/72t2hsbIRCocDDDz+MH//4x1KXRT6mu7sbjz76KKxWKwRBQG5uLhYvXix1WTSEeFE4H3b48GEEBgbiueeeY1ghj6ivr4fZbMaECRPQ3t6OefPm4e2330ZiYqLUpZEPEUURHR0dCAoKgs1mwyOPPIJly5YhIyND6tJoiHAZyIdNmjQJYWFhUpdBPiw6OhoTJkwAAAQHByMhIQEmk0niqsjXKBQKBAUFAQDsdjvsdjsUCoXEVdFQYlghokFRU1ODkpISpKenS10K+SBBEDB37lzcdddduOuuu/hzNswwrBDRD2axWLB48WI8//zzCA4Olroc8kEqlQpbt27F3r17UVxcjLKyMqlLoiHEsEJEP4jNZsPixYvxwAMPYMaMGVKXQz4uNDQUkydPxr59+6QuhYYQwwoR3TRRFLFs2TIkJCTgJz/5idTlkI9qampCa2srAKCrqwsHDhxAQkKCxFXRUOLZQD7s2WefxaFDh9Dc3IyoqCj86le/woIFC6Qui3zIkSNH8Oijj8JoNEKpdH72efbZZzFt2jSJKyNfcvr0aSxZsgSCIEAURcycORNPP/201GXREGJYISIiIlnjMhARERHJGsMKERERyRrDChEREckawwoRERHJGsMKERERyRrvukxELuPGjYPRaIQgCEhISMArr7wCrVbb59y33noLgYGBeOKJJ4a4SiIabthZISKXgIAAbN26Fdu3b4efnx8++OADqUsiImJnhYj6NnHiRJSWlgIAPv30U2zYsAEKhQLJycl49dVXe839+9//jg8//BA2mw3x8fFYtWoVtFotvvjiC7z99ttQKpUICQnB3/72N5SXl2Pp0qWw2WxwOBx46623MHr0aAm+QyLyFgwrROTGbrejqKgI99xzD8rLy/HOO+/g/fffR2RkJFpaWtzm5+Tk4OGHHwYAvPHGG9iyZQsWLlyINWvWYMOGDTAYDK7LpX/wwQdYtGgR5syZA6vVCofDMaTfGxF5H4YVInLp6urC3LlzATg7K/Pnz8eHH36ImTNnIjIyEgAQHh7u9nXl5eV488030dbWBovFgqlTpwIAfvSjH2HJkiWYNWsWcnJyAAAZGRlYu3Yt6urqMGPGDHZViOi6GFaIyOXynpWBWrJkCdasWYOUlBR88sknOHToEABg+fLlOHbsGPbs2YN58+bh448/xgMPPID09HTs2bMHTz75JH7/+9/jzjvvHOxvhYh8CDfYElG/pkyZgn/+859obm4GgD6XgSwWC/R6PWw2Gz777DPX31dVVSE9PR3PPPMMIiIiUFdXh+rqasTFxWHRokXIyspy7YshIroWdlaIqF9JSUn4xS9+gYULF0KpVGL8+PFYuXJlrznPPPMMFixYgMjISKSnp8NisQAAVq1ahfPnz0MURUyZMgUpKSl49913sXXrVqjVauh0Ojz11FNSfFtE5EV412UiIiKSNS4DERERkawxrBAREZGsMawQERGRrDGsEBERkawxrBAREZGsMawQERGRrDGsEBERkaz9L5E35hBOocfCAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Scaling the Age feature\n",
        "data = [train_df, test_df]\n",
        "for dataset in data:\n",
        "    dataset['Age'] = dataset['Age'].astype(int)\n",
        "    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0\n",
        "    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1\n",
        "    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2\n",
        "    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3\n",
        "    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4\n",
        "    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5\n",
        "    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6\n",
        "    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6\n",
        "\n",
        "# let's see how it's distributed \n",
        "train_df['Age'].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S9Ge9-XtRaz5",
        "outputId": "6146ae3e-48c0-4eea-b52d-a9adfde814e5"
      },
      "id": "S9Ge9-XtRaz5",
      "execution_count": 323,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3    283\n",
              "6    148\n",
              "4    126\n",
              "5    103\n",
              "2     92\n",
              "1     71\n",
              "0     68\n",
              "Name: Age, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 323
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = [train_df, test_df]\n",
        "\n",
        "for dataset in data:\n",
        "    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0\n",
        "    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1\n",
        "    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2\n",
        "    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3\n",
        "    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4\n",
        "    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5\n",
        "    dataset['Fare'] = dataset['Fare'].astype(int)"
      ],
      "metadata": {
        "id": "_U9CIxzmRx1o"
      },
      "id": "_U9CIxzmRx1o",
      "execution_count": 324,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "oBBLwrxFRzgS",
        "outputId": "97c9bc32-3d54-45ed-b335-a24e4553ad4b"
      },
      "id": "oBBLwrxFRzgS",
      "execution_count": 325,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Survived  Pclass  Sex  Age  SibSp  Parch  Fare  Embarked  FamSize  Child  \\\n",
              "0         0       3    1    2      1      0     0         2        1      0   \n",
              "1         1       1    0    5      1      0     3         0        1      0   \n",
              "2         1       3    0    3      0      0     0         2        0      0   \n",
              "3         1       1    0    5      1      0     3         2        1      0   \n",
              "4         0       3    1    5      0      0     1         2        0      0   \n",
              "\n",
              "   Alone  SmallFam  LargeFam  LowFare  \n",
              "0      0         1         0        1  \n",
              "1      0         1         0        0  \n",
              "2      1         0         0        1  \n",
              "3      0         1         0        0  \n",
              "4      1         0         0        1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b4663d46-a5c1-4ced-825d-0904b4126a77\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "      <th>FamSize</th>\n",
              "      <th>Child</th>\n",
              "      <th>Alone</th>\n",
              "      <th>SmallFam</th>\n",
              "      <th>LargeFam</th>\n",
              "      <th>LowFare</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>5</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>5</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b4663d46-a5c1-4ced-825d-0904b4126a77')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b4663d46-a5c1-4ced-825d-0904b4126a77 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b4663d46-a5c1-4ced-825d-0904b4126a77');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 325
        }
      ]
    },
    {
      "cell_type": "markdown",
      "id": "b8df3939",
      "metadata": {
        "id": "b8df3939"
      },
      "source": [
        "# FEATURE ENGINEERING "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 326,
      "id": "eca781ee",
      "metadata": {
        "id": "eca781ee"
      },
      "outputs": [],
      "source": [
        "# Function which checks if an individual has age less than 5 years and assigns it value 1 \n",
        "# A new feature 'Child' is created in train and test data sets \n",
        "\n",
        "def is_child(x):\n",
        "    if int(x) <= 5:\n",
        "        return 1\n",
        "    else:\n",
        "        return 0\n",
        "\n",
        "train_df['Child'] = train_df.Age.apply(is_child)\n",
        "test_df['Child'] = test_df.Age.apply(is_child)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 327,
      "id": "91e85d95",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "id": "91e85d95",
        "outputId": "11f8d456-21ad-4a0d-a898-07f03a44e48a"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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qqqqUlpamm677Ta1atWqQQcYNGiQBg0apNWrV2vGjBmaMmVKvcuXllY22GOHhjbszwJI0tGj5Y09Qh1s67BCQ23r7rZPj4f4XnvtNT333HMqKipSenq6HnzwQb377rseH9DhcKioqMh1ubi4WA6Hw+3yvXv31saNGz2uFwDgG87qOaiLL75Yo0eP1pgxY/Thhx/qqaee0q233qoNGza4vU9MTIz27t2rffv2qaqqSrm5uYqPj6+1zN69e13fv/XWW4qIiPh5PwUA4Lzj8RDfrl27tGzZMm3atEndunXTzJkzFRUVpeLiYt1+++3q1avXmVdstysrK0tDhw6V0+lUv379FBkZqWnTpik6Olo9evTQwoULtWXLFtntdrVu3drj4T0AgO/wGKinnnpKGRkZeuCBB9S8eXPX9Q6HQ6NGjar3vnFxcYqLi6t13en3eeyxx851XgCAj/B4iK9nz57q27dvrTiderVd3759rZsMAODTPAZq5cqVda5bvny5JcPAPDY//1pfAcBb3B7iy8nJUU5Ojvbv36977rnHdX1ZWZnatGnjleHQ+ILCb1HZd1vU8qIbG3sUAD7GbaCuvfZahYaG6siRI/rDH/7gur5ly5Y/6427aJoCQiIVEBLZ2GMA8EFuA3XRRRfpoosu0uLFi705DwAAkuoJVGZmphYtWqRrr7221jnyampqZLPZtH37dq8MCADwTW4DtWjRIknSxx9/7LVhAAA4xW2gjh49Wu8dg4ODG3wYAABOcRuo9PR02Ww21dTU1LnNZrPV+egMAAAakttA5efne3MOAABqcRuor776Sp06ddKOHTvOeHtUVJRlQwEA4DZQ8+bN05NPPqnJkyfXuc1ms2n+/PmWDgYA8G1uA/Xkk09KkhYsWOC1YQAAOMXj2cwrKyv12muv6aOPPpLNZtN1112nzMxMBQQEeGM+AICP8niy2HHjxmnPnj264447NGjQIBUWFmrs2LHemA0A4MM87kHt2bNHa9ascV2OjY1VcnKypUMBAOBxD+rKK6/UJ5984rr86aefKjo62tKhAABwuweVkpIiSTpx4oRuv/12/epXv5Ikff/997r00ku9Mx0AwGe5DdTMmTO9OQcAALXU+3Ebpzt8+LAqKystHwgAAOksXiSRl5enKVOm6ODBgwoJCdH333+vTp06KTc31xvzAQB8lMcXSUybNk2L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CgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEayNFCbN29WYmKiEhISNGvWrDq3V1VVafTo0UpISFD//v21f/9+K8cBADQhlgXK6XRqwoQJys7OVm5urnJyclRYWFhrmSVLlqh169Z64403dNddd+m5556zahwAQBNjWaAKCgoUERGhsLAw+fv7q3fv3srLy6u1TH5+vtLS0iRJiYmJ2rJli2pqaqwaCQDQhNitWnFxcbHat2/vuuxwOFRQUFBnmQ4dOpwcxG5Xq1atdOTIEYWEhLhdb2hoqwad87UJtzTo+oCG3kYbyuVj5jX2CDjPWL2t8yIJAICRLAuUw+FQUVGR63JxcbEcDkedZQ4cOCBJOnHihEpKStS2bVurRgIANCGWBSomJkZ79+7Vvn37VFVVpdzcXMXHx9daJj4+XsuXL5ckrV+/XrGxsbLZbFaNBABoQmw1Fr4qYdOmTZo4caKcTqf69eun4cOHa9q0aYqOjlaPHj1UWVmpsWPH6osvvlCbNm00depUhYWFWTUOAKAJsTRQAAD8XLxIAgBgJAIFADASgTpPFBUVafjw4erVq5d69uypp556SlVVVZY8Vnx8vFJSUpSamqr09HRLHgM4Zf/+/erTp0+t61544QXNmTPnnNYzePBgffbZZz97jmXLlik2NlapqalKTU3VkiVLfva6cHYse6MuvKempkYjR45UZmamZsyYIafTqccff1xTp07VQw899IvWfeLECdntdTeTV155pd43VLvjdDrl5+f3i2YCvOFM22pycrKysrLOeV3u/o5QP/agzgPvv/++AgIC1K9fP0mSn5+fHn30US1btkwVFRUaMGCA9uzZ41r+1P8ky8vL9cgjjygjI0N9+/bVxo0bJZ38n+I999yjIUOG6K677jqrGb799lvXaaskae/eva7L8fHxevbZZ5WWlqZ169Zp/vz5Sk5OVkpKiu6///4G+i3AVw0ePFjPPvusMjIylJiYqG3btkmSjh8/rvvvv19JSUkaMWKEjh8/7rrPO++8o4EDByotLU333XefysrKJNXdVs/GuHHjXH87kvTggw9q48aNdf6ODh48qEGDBik1NVV9+vRxzQn3SPp5YM+ePYqKiqp1XVBQkDp06KBvvvlGycnJWrt2rSIjI3Xw4EEdPHhQMTEx+sc//qHY2FhNmjRJP/74o/r3769u3bpJknbu3KlVq1YpODj4jI/5xz/+UTabTQMHDtTAgQMVHh6uoKAgffHFF7riiiu0bNmyWof/goODXe956969u/Lz8+Xv768ff/zRot8KfInT6dTSpUu1adMmvfjii5o3b54WLVqk5s2ba+3atdq1a5dre/zhhx80Y8YMzZ07Vy1atNCsWbM0d+5cjRw5UlLtbfW/bdiwQVu3btUll1yiRx55RB06dFBGRobmzZunnj17qqSkRB9//LGmTJmiVatW1fo7+te//qXu3btr+PDhcjqdqqio8Nrvp6liD8oHJCUlaf369ZKktWvX6tZbb5V08n+Rs2fPVmpqqgYPHqzKykrXmT1uuukmt3FatGiRli9frtmzZ+vVV1/V1q1bJUn9+/fX66+/LqfTqTVr1tR63iA5Odn1fefOnTVmzBitXLmSw33wyN2b90+/PiEhQZIUFRWl7777TpK0detW3XbbbZKkLl26qHPnzpKkTz/9VIWFhcrMzFRqaqpWrFih77//3rWu07fV0/3ud79Tfn6+Vq9erW7durkOn99www365ptv9MMPPygnJ0eJiYmuw3mn/x3FxMRo2bJleuGFF7R7924FBQX97N+JryBQ54HLLrtMO3bsqHVdaWmpDhw4oIiICDkcDgUHB2vXrl1au3atkpKSXMv985//1MqVK7Vy5Uq99dZb6tSpkyQpMDDQ7eOdOmXVBRdcoISEBNdJgBMTE/X222/rzTffVFRUVK3TVp2+vlmzZun3v/+9du7cqYyMDJ04ceKX/xJw3goODtaxY8dqXXfs2LFa25e/v78kqVmzZnI6nfWur6amRjfddJNru1+zZo0mTpzout3dtt+2bVvX4/Tv37/W31xqaqpWrVqlZcuWuQ61//e6rr/+ei1cuFAOh0MPP/ywVqxY4elH93kE6jxw4403qqKiwrXBO51OTZ48WWlpaa4/kOTkZGVnZ6ukpERdunSRdPJQ28KFC10fcbJz506Pj1VeXq7S0lLX9++++64iIyMlSQEBAerevbueeOIJt6/uq66u1oEDBxQbG6sxY8aopKRE5eXlv+wXgPNay5YtFRoaqi1btkiSjh49qrffflvXXXddvfe7/vrrlZOTI0navXu3vvzyS0nSNddco+3bt+ubb76RdHI7/vrrrz3OcfDgQdf3+fn5rv/MSVJ6erpeeeUVSSf/w3gm3333nS688EINGDCgTuBwZjwHdR6w2Wx66aWXNH78eE2fPl3V1dWKi4vTAw884FomMTFRTz/9tO69917Xdffee68mTpyo2267TdXV1erYsaNefvnleh/r8OHDGjFihKSTIezTp49uvvlm1+0pKSl644031L179zPe3+l0auzYsSotLVVNTY2GDBmi1q1b/5IfHz7gmWee0fjx4zV58mRJ0ogRIxQeHl7vfTIzM/XII48oKSlJnTp1cj1PGxISokmTJumBBx5wvRVj9OjRuuSSS+pd34IFC5Sfny8/Pz+1adNGkyZNct124YUX6tJLL1XPnj3d3v/DDz/UnDlzZLfb1aJFC02ZMuWsfnZfxqmO0KDmzJmjkpISjR49urFHAbymoqJCKSkpWr58uVq1MvPzwJoiDvGhwYwYMUIrVqzQkCFDGnsUwGvee+89JScn64477iBODYw9KACAkdiDAgAYiUABAIxEoAAARiJQgBcdOnRI999/v3r27Kn09HQNGzZMixcv1p///OczLv/Xv/5VhYWFkk6eJ+6HH36os8zPObM30BTwPijAS06ddb5v376aOnWqJGnXrl3Ky8tze5+nn37aW+MBxmEPCvCS999/X3a7XZmZma7runTpoq5du6q8vFz33Xefbr31Vj344IOus3u4+wyjGTNmKDExUZmZmWd1FgSgKWIPCvCSM511/pSdO3cqNzdX7dq1U2Zmpj766CN17dr1jMt+/vnnWrNmjVasWCGn06m0tDS36wWaMvagAANcddVVat++vZo1a6YuXbq4zsh9Jtu2bVPPnj0VGBiooKAgxcfHe3FSwHsIFOAlkZGRbk8Qeuos2dLJD5z0dEZuwBcQKMBLYmNjVVVVpcWLF7uu27Vr1zl/sur111+vjRs36vjx4yotLdWbb77Z0KMCRuA5KMBLbDabXnzxRU2cOFGzZ89WQECALrroonrPgH0mUVFRSk5OVmpqqkJCQhQTE2PRxEDj4lx8AAAjcYgPAGAkAgUAMBKBAgAYiUABAIxEoAAARiJQAAAjESgAgJH+F9K2BnmvFkqcAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "g = sns.catplot(x=\"Child\", y=\"Survived\", data=train_df, height=6,kind=\"bar\", palette=\"muted\")\n",
        "g.despine(left=True)\n",
        "g.set_xticklabels(['Over 5yrs', 'Under 5yrs'])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()\n",
        "\n",
        "# Children have higher probability of survival"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 298,
      "id": "4e526430",
      "metadata": {
        "id": "4e526430"
      },
      "outputs": [],
      "source": [
        "# SIBSP & PARCH\n",
        "\n",
        "train_df['FamSize'] = train_df.SibSp + train_df.Parch\n",
        "test_df['FamSize'] = test_df.SibSp + test_df.Parch"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 299,
      "id": "82c98382",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "id": "82c98382",
        "outputId": "3d457ef0-2bdf-4aba-d503-a0fa502d0f39"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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AID6CHqx2NzcXH3xxRe64YYbdP3112v79u2aPHlyKLIBAKJY0BHUF198oXXr1vnne/furcGDB9saCgCAoCOoX/ziF/rnP//pn//Xv/6l1NRUW0MBAFDnCCorK0uSVFNTo1GjRumMM86QJH377bc6++yzQ5MOABC16iyoBQsWhDIHAAABTni7jR+rqKiQx+OxPRAAAFI9TpIoKCjQI488ot27dysxMVHffvutOnfurPz8/FDkAwBEqaAnScydO1fPPfeczjzzTBUWFmrp0qU6//zzQ5ENABDFghZUbGys2rRpo9raWtXW1qp379765JNPQpENABDFgu7ii4+PV1VVlS666CLde++9SkxMVPPmzUORDQAQxYKOoObNm6emTZvq/vvv1xVXXKGOHTtq/vz5ocgGAIhiQUdQzZs31549e1RSUqKEhARdfvnlatOmTSiyAQCiWNAR1KpVqzRy5Ei99dZbeuONN3Tddddp9erVocgGAIhiQUdQixcv1ksvveQfNe3du1ejRo1Sdna27eEAANEr6AiqTZs2atGihX++RYsW7OLDccXExQRMAeDnqHMEtWTJEklSx44dde211yo9PV0Oh0MFBQXq3r17yAJCahLrCJia6owrk1W+cbeSL2sX7igATgF1FlRVVZWkIwXVsWNH/+Pp6en2p0KArK4JevvrSvU7q1W4o5xQQtd4JXSND3cMAKeIOgtq3LhxAfNHC+vHu/sQGj3bNVPPds3CHQMAQiroSRLbtm1Tbm6u9u/fL+nIMalHHnlEXbt2tT0cACB6BS2ovLw8TZkyRb1795Ykffjhh5o+fbr+/ve/2x4OABC9gp7FV11d7S8nSbr00ktVXV1taygAAIKOoFJSUvTkk09qyJAhkqRXXnlFKSkptgcDAES3oCOo2bNna+/evbrrrrs0fvx47d27V7Nnzw5FNgBAFDvhCMrn82ncuHFavnx5qPIAACApyAgqJiZGjRo1UmVlZajyAAAgqZ5XM8/KylKfPn0C7gM1bdo0W4MBAKJb0IIaMGCABgwYEIosAAD4BS2oYcOGyev16quvvpLD4dBZZ52luLi4UGQDAESxoAW1fv165eXlqWPHjrIsS2VlZZoxY4bS0tJCkQ8AEKWCFtRDDz2kZcuWqVOnTpKkHTt2KCcnh4ICANgq6OegWrRo4S8n6cgHd7lgLADAbkFHUKmpqbrjjjs0aNAgORwOvf766+rZs6fefPNNSeIECgCALYIWlNfr1WmnnabNmzdLkhITE+XxePTOO+9IoqAAAPao1zEoAABCLegxKAAAwoGCAgAYiYICABipzmNQS5YsOeGKt95660kPAwDAUXUWVFVVVShzAAAQoM6CGjduXChzAAAQIOhp5h6PR6tXr9YXX3whj8fjf5zTzwEAdgp6ksTkyZO1Z88e/eMf/9All1wit9vNpY4AALYLWlA7duzQxIkT1axZMw0bNkwLFy5USUlJKLIBAKJY0IKKjT2yFzA+Pl7btm1TZWWlKioqbA8GAIhuQY9BXXfdddq/f78mTJigsWPHqrq6WhMmTAhFNgBAFAs6gho+fLgSEhJ0ySWXqKCgQBs3btSoUaPqtfGioiJlZGSof//+WrRoUZ3LvfHGG+revbv+/e9/1z85AOCUFrSg0tPTNX36dG3cuFGWZdV7wz6fTzNnztTixYuVn5+vtWvXavv27ccsd+DAAS1btkznn3/+T0sOADilBS2o1157TZdddpmeeeYZuVwuzZw5U1u2bAm64ZKSEnXq1EkpKSmKi4tTZmamCgoKjllu7ty5uuOOO9SkSZOGfQXAKai4eItmzJiq4uLg7zXgVBX0GFSzZs00ePBgDR48WPv379esWbN044036rPPPjvhem63W8nJyf55p9N5zNl/W7duVXl5ua688ko9/fTT9QocE+NQ69bN67UsjhUJ37tIyCjZm/PFF/+u7du36/Bhj1yuvra9TjSIiXH4p5Hyu4UjghaUJG3atEnr1q3Thg0blJqaqjlz5vzsF66trdXDDz/8kz/w6/NZ2rev+me/fqglJbUKdwRJOuH3LhIySpGT8+c4cKDaP43E33eT+HyWf8r30kx1vaeDFpTL5dK5556rQYMGKTc3V82b1+9/IE6nU+Xl5f55t9stp9Ppn6+qqtK2bdt00003SZL27NmjsWPHav78+erZs2e9XgMAcOoKWlCvvPKKWrZs+ZM33LNnT5WWlmrnzp1yOp3Kz8/Xn/70J//zrVq10ocffuifv/HGG5Wbm0s5AQAknaCgnnrqKd1xxx164okn5HA4jnl+2rRpJ95wbKzy8vI0ZswY+Xw+jRgxQl27dtXcuXOVmpqq9PT0n58eAHDKqrOgOnfuLElKTU1t8MbT0tKUlpYW8FhdH/Jdvnx5g18HAHDqqbOgXC6XJKlbt27q0aNHyAIBACDV4xjUww8/rO+++04ZGRkaPHiwunXrFopcAIAoF7Sgli9frj179ui1115TXl6eqqqqNGjQIN15552hyAcAiFJBryQhSUlJSbrppps0Y8YMnXPOOZo3b57duQAAUS7oCOrLL7/UunXr9Oabb6p169YaNGiQpkyZEopsAIAoFrSg7r//fg0ePFiLFy8O+KAtAAB2OmFB+Xw+dejQQTfffHOo8gAAICnIMaiYmBjt2rVLXq83VHkAAJBUj118HTp00OjRo+VyuQKuw3frrbfaGgwAEN2CFlTHjh3VsWNHWZalqqqqUGQCACB4QY0bNy4UOQAACBC0oG688cbjXix22bJltgQCAECqR0H97ne/8//b4/HozTffVExMjK2hAAAIWlD//2rmF154obKzs20LBACAVI+C2rdvn//ftbW12rp1qyorK20NBQBA0IIaPny4HA6HLMtSbGysOnTooFmzZoUiGwAgigUtqMLCwlDkAAAgQNCrmb/22ms6cOCAJGnevHkaN26ctm7danswAEB0C1pQ8+bNU8uWLbVlyxZt3LhR2dnZ+sMf/hCCaKFRXLxFM2ZMVXHxlnBHAQD8SNCCOnpK+fr163Xttdfqyiuv1OHDh20PFiqrVq3UZ59t1apVK8MdBQDwI0ELyul0Ki8vT+vWrVNaWpq8Xq9qa2tDkS0kDh48FDAFAJghaEHNmTNHl19+uZ5++mnFx8dr3759ys3NDUU2AEAUC3oWX7NmzTRgwAD/fLt27dSuXTtbQwHAUW0Smik2LuifqjrFxDj806SkVg3aRo23Rnv3H2xwBjRMw3/qABACsXGx+tdjbzd4fe/eav+0ods5f3K/Br8+Gi7oLj4AAMKBggIAGImCAgAYiYICABiJggIAGImCAgAYiYICABiJggIAGImCAgAYiYICABiJggIAGCnir8XXKr6ZmjYJ74UkD3lqVPkDF5IEgJMp4guqaZNY/Srv3Qav/13FkWIprzjY4O2snHmlKhucAABwPOziAwAYiYICABiJggIAGImCAgAYiYICABiJggIAGImCAgAYiYICABiJggIAGImCAtAgxcVbNGPGVBUXbwl3FJyiIv5SRwDCY9Wqlfr666906NBBXXDBReGOg1MQIygADXLw4KGAKXCyRX1BOWLiAqYAADNEfUG17HilGsd3UsuOV4Y7CgDgR6L+GFSTxK5qktg13DEAAP9P1I+gAABmoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaytaCKioqUkZGh/v37a9GiRcc8v2TJEg0ePFhZWVm6+eab9c0339gZBwAQQWwrKJ/Pp5kzZ2rx4sXKz8/X2rVrtX379oBlzj33XL3wwgt69dVXlZGRoccee8yuOACACGNbQZWUlKhTp05KSUlRXFycMjMzVVBQELBM79691axZM0nSL3/5S5WXl9sVBwAQYWLt2rDb7VZycrJ/3ul0qqSkpM7lV69erb59+wbdbkyMQ61bNz8pGU8mEzMdTyTkjISM0olzOiTFNo5p8LZjYhz+aVJSqwZto+awT1aDEwT344yR8jP7OaLhazSNbQX1U6xZs0affPKJVqxYEXRZn8/Svn3V/vmGvnlPth9nOp5IyBkJGaXIyJmU1EpP5q5u8Lb3f3fAP23odn77aLb27KlscIZgfD7LPw32M/s5IuHnjZ+nrp+xbQXldDoDdtm53W45nc5jlnv//fe1YMECrVixQnFxcXbFAQBEGNuOQfXs2VOlpaXauXOnvF6v8vPz5XK5Apb59NNPlZeXp/nz56tt27Z2RQEARCDbRlCxsbHKy8vTmDFj5PP5NGLECHXt2lVz585Vamqq0tPT9eijj6q6uloTJkyQJJ1++ulasGCBXZEAABHE1mNQaWlpSktLC3jsaBlJ0tKlS+18eQBABONKEgAAI1FQAAAjUVAAACNRUAAAI1FQAAAjUVAAACNRUAAAI1FQAAAjUVAAACNRUAAAI1FQAAAjUVAAACNRUAAAI1FQAAAjUVAAACNRUAAAI1FQAAAjUVAAACNRUAAAI1FQAAAjUVAAACPFhjsAgPBJiG+suCZNG7RuTIzDP01KatXgDF7PIe3/4XCD18epi4IColhck6Z64PaMBq37vbvm/6bfNHgbkjT96TckUVA4Frv4AABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioACc0prENgmYInJQUABOaQO7XKHOiR01sMsV4Y6Cnyg23AEAwE6/aNdFv2jXJdwx0ACMoAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGoqAAAEaioAAARqKgAABGsrWgioqKlJGRof79+2vRokXHPO/1ejVx4kT1799fI0eOVFlZmZ1xAAARxLaC8vl8mjlzphYvXqz8/HytXbtW27dvD1hm1apVio+P11tvvaVbbrlFf/zjH+2KAwCIMLYVVElJiTp16qSUlBTFxcUpMzNTBQUFAcsUFhZq2LBhkqSMjAxt3LhRlmXZFQkAEEEclk2N8Prrr2vDhg2aNWuWJOnll19WSUmJ8vLy/MtcffXVWrx4sZKTkyVJ/fr10/PPP6/ExEQ7IgEAIggnSQAAjGRbQTmdTpWXl/vn3W63nE7nMcvs2rVLklRTU6PKykq1adPGrkgAgAhiW0H17NlTpaWl2rlzp7xer/Lz8+VyuQKWcblceumllyRJb7zxhnr37i2Hw2FXJABABLHtGJQkrV+/XrNnz5bP59OIESM0duxYzZ07V6mpqUpPT5fH49HkyZP12WefKSEhQU888YRSUlLsigMAiCC2FhQAAA3FSRIAACNRUAAAI8WGO0A4FRUVadasWaqtrdXIkSOVk5MT7kjHuO+++/Tuu++qbdu2Wrt2bbjjHNeuXbuUm5uriooKORwOXXvttbr55pvDHesYHo9H119/vbxer3w+nzIyMjR+/Phwxzquo8dtnU6nFi5cGO44x+VyudSiRQs1atRIMTExevHFF8Md6Rg//PCDpk2bpm3btsnhcGj27Nnq1atXuGMF+Oqrr3T33Xf753fu3Knx48frlltuCV+o/3O8vz/79u3T3XffrW+++Ubt27fXnDlzlJCQYE8AK0rV1NRY6enp1o4dOyyPx2NlZWVZX3zxRbhjHWPTpk3WJ598YmVmZoY7Sp3cbrf1ySefWJZlWZWVldaAAQOM/F7W1tZaBw4csCzLsrxer5WdnW19/PHHYU51fH/961+tSZMmWTk5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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "g = sns.catplot(x=\"FamSize\", y=\"Survived\", data=train_df, height=6, kind=\"bar\", palette=\"muted\")\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()\n",
        "# This shows that the survival probability for an individual with less or equal 3 relatives are high"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 300,
      "id": "273b2aad",
      "metadata": {
        "id": "273b2aad"
      },
      "outputs": [],
      "source": [
        "def is_alone(x):\n",
        "    if int(x) == 0:\n",
        "        return 1\n",
        "    else:\n",
        "        return 0\n",
        "\n",
        "train_df['Alone'] = train_df.FamSize.apply(is_alone)\n",
        "test_df['Alone'] = test_df.FamSize.apply(is_alone)\n",
        "\n",
        "def is_small_fam(x):\n",
        "    if int(x) == 0:\n",
        "        return 0\n",
        "    elif int(x) >= 4:\n",
        "        return 0\n",
        "    else:\n",
        "        return 1\n",
        "\n",
        "train_df['SmallFam'] = train_df.FamSize.apply(is_small_fam)\n",
        "test_df['SmallFam'] = test_df.FamSize.apply(is_small_fam)\n",
        "\n",
        "def is_large_fam(x):\n",
        "    if int(x) >= 4:\n",
        "        return 1\n",
        "    else:\n",
        "        return 0\n",
        "\n",
        "train_df['LargeFam'] = train_df.FamSize.apply(is_large_fam)\n",
        "test_df['LargeFam'] = test_df.FamSize.apply(is_large_fam)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 301,
      "id": "a29f70dd",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "id": "a29f70dd",
        "outputId": "bd5f9286-0f78-461a-878e-b744d35cd6c2"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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9Ag4wVj9WeciCQCAkSwLlMPhUENDg3/Z4/HI4XAcsc6ePXskSQcPHlRTU5P69etn1UgAgC7EskClpaWprq5O9fX1amtrk8vlktPpDFjH6XRq7dq1kqSNGzdq9OjRstlsVo0EAOhCbD4Lr0rYvHmzHnnkEXm9Xk2ePFk33XSTSktLlZqaqoyMDLW2turOO+/Uhx9+qL59++qJJ55QUlKSVeMAALoQSwMFAMB/i4skAABGIlAAACMRKAQYNmyY5s2b519eunSpnnrqqQhOBISOz+dTQUGBNm/e7H9uw4YNuu666yI4FTpCoBAgNjZWmzZt0tdffx3pUYCQs9lsmjNnjubNm6fW1lY1NzfriSee0AMPPBDp0XAUXCSBACNGjNCNN96olpYW3XbbbVq6dKlaWlp0yy23aPfu3br33nv9d/soKSnRySefHOmRgeM2f/58xcXFqaWlRXFxcfrss8+0c+dOHTx4UIWFhRo7dqx27type+65RwcOHFB7e7ueeuopnXrqqZEePaqwB4UjTJ8+XRUVFWpqagp4fu7cucrLy1NFRYVyc3M1d+7cCE0I/DCFhYWqqKjQ66+/rtbWVo0ePVqrV69WeXm5HnvsMbW0tOj555/XjBkztG7dOr344osBt25DeFh2Lz50Xb1799bEiRNVXl6uHj16+J9/9913/eejJk6cqMceeyxSIwI/SFxcnLKzsxUXF6cNGzbo1Vdf1Z///GdJUmtrq/bs2aPhw4dr0aJFamhoUGZmJntPEUCgcFRXX3218vPzlZ+fH+lRAEt069ZN3bodOoj0+9//XqeddlrA60OGDNG5556r1157TTfccIPmzJmj888/PxKjRi0O8eGoEhIS9Mtf/lKrV6/2PzdixAi5XC5JUkVFhUaNGhWp8YCQueiii7RixQr/n/r54IMPJEn19fVKSkrSjBkzlJGRoY8++iiSY0YlAoUOXXvttdq3b59/+f7779eaNWuUm5urdevW6Te/+U0EpwNC4+abb9bBgwd16aWXKicnR6WlpZIOXX4+YcIETZw4UTt27NCkSZMiPGn04So+AICR2IMCABiJQAEAjESgAABGIlAAACMRKACAkQgUECZ//etfNWzYMH388ceSpN27d2vChAkRngowF4ECwqSyslIjR470/2dnAJ0jUEAYNDc3a+vWrXr44YePGqjW1lbdc889ys3N1aRJk/Tmm29KktasWaPCwkJdd911yszM1Pz58/3f88Ybb2jatGnKy8vTr3/9azU3N4ft5wHCgUABYVBVVaWLL75YgwcPVr9+/fT+++8HvP7ss89KOnQLqccff1x33323WltbJUkffvihnnzySVVUVGjDhg3as2ePvv76ay1cuFBlZWVau3atUlNTVVZWFvafC7ASN4sFwsDlcmnGjBmSpOzsbLlcLk2fPt3/+tatW3XllVdKOnST0pNPPlm7du2SJJ1//vmKj4/3v/bZZ5+pqalJtbW1KigokCQdOHBAw4cPD+ePBFiOQAEWa2xs1JtvvqkdO3bIZrPJ6/XKZrPpiiuuOKbvj42N9T+OiYmR1+uVz+fThRdeqN/97ndWjQ1EHIf4AItt3LhREydO1Kuvvqrq6mpt3rxZgwYNUkNDg3+dUaNGqaKiQpK0a9cu7dmz54g///B9w4cP17Zt2/Tvf/9bktTS0uLf4wJOFAQKsFhlZaXGjh0b8FxmZqaefvpp//IVV1whn8+n3Nxc3XbbbSopKQnYc/pP/fv3V0lJiW6//Xbl5uZq2rRp+uSTTyz7GYBI4G7mAAAjsQcFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEj/B43nHCkHo55AAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "g = sns.catplot(x=\"Alone\", y=\"Survived\", data=train_df, height=6, \n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "g.set_xticklabels(['No', 'Yes'])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "MXRRWkNKIm8m",
        "outputId": "f423dc9e-66f1-411f-c313-26c7c9026753"
      },
      "id": "MXRRWkNKIm8m",
      "execution_count": 290,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     PassengerId  Survived  Pclass  Sex  Age  SibSp  Parch  Fare  Embarked  \\\n",
              "0              1         0       3    1   22      1      0     7         2   \n",
              "1              2         1       1    0   38      1      0    71         0   \n",
              "2              3         1       3    0   26      0      0     7         2   \n",
              "3              4         1       1    0   35      1      0    53         2   \n",
              "4              5         0       3    1   35      0      0     8         2   \n",
              "..           ...       ...     ...  ...  ...    ...    ...   ...       ...   \n",
              "886          887         0       2    1   27      0      0    13         2   \n",
              "887          888         1       1    0   19      0      0    30         2   \n",
              "888          889         0       3    0   25      1      2    23         2   \n",
              "889          890         1       1    1   26      0      0    30         0   \n",
              "890          891         0       3    1   32      0      0     7         1   \n",
              "\n",
              "     FamSize  Child  Alone  SmallFam  LargeFam  \n",
              "0          1      0      0         1         0  \n",
              "1          1      0      0         1         0  \n",
              "2          0      0      1         0         0  \n",
              "3          1      0      0         1         0  \n",
              "4          0      0      1         0         0  \n",
              "..       ...    ...    ...       ...       ...  \n",
              "886        0      0      1         0         0  \n",
              "887        0      0      1         0         0  \n",
              "888        3      0      0         1         0  \n",
              "889        0      0      1         0         0  \n",
              "890        0      0      1         0         0  \n",
              "\n",
              "[891 rows x 14 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-29b069e7-5b8d-4fbf-80ef-18481f64d38d\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "      <th>FamSize</th>\n",
              "      <th>Child</th>\n",
              "      <th>Alone</th>\n",
              "      <th>SmallFam</th>\n",
              "      <th>LargeFam</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>22</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>38</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>26</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>35</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>35</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>886</th>\n",
              "      <td>887</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>27</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>13</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>887</th>\n",
              "      <td>888</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>19</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>30</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>888</th>\n",
              "      <td>889</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>25</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>23</td>\n",
              "      <td>2</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>889</th>\n",
              "      <td>890</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>26</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>30</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>890</th>\n",
              "      <td>891</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>32</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>891 rows × 14 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-29b069e7-5b8d-4fbf-80ef-18481f64d38d')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-29b069e7-5b8d-4fbf-80ef-18481f64d38d button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-29b069e7-5b8d-4fbf-80ef-18481f64d38d');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 290
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 304,
      "id": "6caf3bfd",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 865
        },
        "id": "6caf3bfd",
        "outputId": "c65c98bf-330d-48f5-a2dc-6abc26c64e98"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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F6iqqsqqcQAArUy4VRt2uVyKi4vzLtvtdjmdzibXX7FihS6//HK/2w0Lsyk6OrJFZgSswPMTocLq57plgToRxcXF+vjjj7V06VK/67rdHh08WNsijxsbG9Ui2wF+rqWeny2J5zqsYPVrsWWBstvtPofsXC6X7Hb7Uev97W9/07x587R06VJFRERYNQ4AoJWx7D2o5ORkVVRUqLKyUvX19SotLZXD4fBZ55NPPlF+fr7mzp2r0047zapRAACtkGV7UOHh4crPz1dubq7cbrdGjhypxMREFRYWKikpSWlpaXriiSdUW1urCRMmSJK6deumefPmWTUSAKAVsfQ9qNTUVKWmpvrc9lOMJGnRokVWPjwAoBXjShIAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGsjRQ5eXlysjIUHp6uoqKio66f/PmzRo+fLjOOeccvfHGG1aOAgBoZSwLlNvtVkFBgebPn6/S0lKtWbNGO3fu9FmnW7dumjFjhoYOHWrVGACAVircqg07nU4lJCQoPj5ekpSVlaWysjL16dPHu0737t0lSW3acKQRAODLskC5XC7FxcV5l+12u5xO5y/ebliYTdHRkb94O4BVeH4iVFj9XLcsUFZxuz06eLC2RbYVGxvVItsBfq6lnp8tiec6rGD1a7Flx9bsdruqqqq8yy6XS3a73aqHAwCcYiwLVHJysioqKlRZWan6+nqVlpbK4XBY9XAAgFOMZYEKDw9Xfn6+cnNzlZmZqSFDhigxMVGFhYUqKyuTdOREissvv1xvvPGGHnnkEWVlZVk1DgCglbH0PajU1FSlpqb63DZhwgTv9+edd57Ky8utHAEA0EpxfjcAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRCBQAwEgECgBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICRLA1UeXm5MjIylJ6erqKioqPur6+v18SJE5Wenq6rr75au3fvtnIcAEArYlmg3G63CgoKNH/+fJWWlmrNmjXauXOnzzrLly9X586d9dZbb+nGG2/UU089ZdU4AIBWxrJAOZ1OJSQkKD4+XhEREcrKylJZWZnPOhs2bNDw4cMlSRkZGXr//ffl8XisGgkA0IqEW7Vhl8uluLg477LdbpfT6TxqnW7duh0ZJDxcUVFROnDggLp27drkdtu2DVNsbFSLzflSwRUtti1AUos+P1tS3/sWBXsEnGKsfq5zkgQAwEiWBcput6uqqsq77HK5ZLfbj1pnz549kqTGxkZVV1crJibGqpEAAK2IZYFKTk5WRUWFKisrVV9fr9LSUjkcDp91HA6HVq1aJUl68803lZKSIpvNZtVIAIBWxOax8KyEjRs3avr06XK73Ro5cqRuv/12FRYWKikpSWlpaaqrq9P999+vTz/9VF26dNHs2bMVHx9v1TgAgFbE0kABAHCyOEkCAGAkAgUAMBKBgo9+/fpp5syZ3uUFCxbo2WefDeJEQMvxeDwaM2aMNm7c6L1t3bp1uuWWW4I4FZpCoOAjIiJC69ev1/79+4M9CtDibDabpk6dqpkzZ6qurk41NTWaPXu2HnnkkWCPhmPgJAn46N+/v2677TbV1tbq7rvv1oIFC1RbW6s777xTu3fv1pQpU7xX+5gxY4bOOOOMYI8MnLAnnnhCkZGRqq2tVWRkpL766ivt2LFDjY2NysvL06BBg7Rjxw49+OCDamho0OHDh/Xss8+qZ8+ewR49pLAHhaOMHTtWJSUlqq6u9rl92rRpGj58uEpKSpSdna1p06YFaULgl8nLy1NJSYneffdd1dXVKSUlRStWrNDixYv15JNPqra2Vi+//LLGjRun4uJivfbaaz6XbkNgWHYtPrRenTp1Uk5OjhYvXqz27dt7b//www+970fl5OToySefDNaIwC8SGRmpzMxMRUZGat26dXr77bf15z//WZJUV1enPXv26IILLtC8efNUVVWlwYMHs/cUBAQKx3TDDTdoxIgRGjFiRLBHASzRpk0btWlz5CDSH/7wB/Xq1cvn/t69e+v888/XO++8o1tvvVVTp07V//zP/wRj1JDFIT4cU3R0tH77299qxYoV3tv69++v0tJSSVJJSYkGDBgQrPGAFnPppZdq6dKl3o/6+eSTTyRJlZWVio+P17hx45SWlqbPPvssmGOGJAKFJt188806cOCAd/nhhx/WypUrlZ2dreLiYv3ud78L4nRAy7jjjjvU2Nioq666SllZWSosLJR05PTzoUOHKicnR9u3b9ewYcOCPGno4Sw+AICR2IMCABiJQAEAjESgAABGIlAAACMRKACAkQgU8AvMnTtXWVlZys7OVk5Ojj766KNfvM3+/ftLknbv3q2hQ4dKkjZt2qQLL7xQOTk5ysnJ0Y033viLHwcwHVeSAE7Shx9+qHfeeUerVq1SRESE9u/fr4aGBsseb8CAAXr++ect2z5gGgIFnKS9e/cqJiZGERERkqSuXbtKkhwOh7KyslReXq6wsDA99thjevrpp7Vr1y7dcsstGjNmjGpqanTHHXfo+++/V2NjoyZMmKBBgwad0OM7nU49/vjjqqurU/v27TV9+nT16tVLK1eu1F/+8hcdOnRIu3bt0s0336yGhgYVFxcrIiJCRUVFio6ObvG/B9DSOMQHnKRLLrlEe/bsUUZGhh599FF98MEH3vu6deum4uJiDRgwQJMnT1ZhYaFeffVV78V227Vrp+eee06rVq3SCy+8oFmzZsnf/8xv2bLFe4hv7ty56tWrl1588UWtXr1ad911l2bPnu1dd8eOHXr22We1YsUKzZ49W+3bt9fq1at1wQUXaPXq1db8QYAWxh4UcJI6duyolStXasuWLdq0aZPuvvtu3XvvvZKktLQ0SVLfvn1VW1urTp06STrygZDff/+9OnTooKefflqbN29WmzZt5HK59O233yo2NrbJx/vvQ3x79uzRAw88oF27dslms/kcXhw4cKD3MaOiouRwOLzzcE05tBYECvgFwsLCNHDgQA0cOFB9+/b17p20bdtW0pErZv90CPCn5cbGRpWUlGj//v1auXKl2rZtK4fDobq6uhN67MLCQg0cOFDPPfecdu/erXHjxnnv++/H/Pk8brf7pH9fIJA4xAecpC+++EIVFRXe5U8//bTZnzBcXV2t0047TW3bttXf//53ffXVVyf8+NXV1bLb7ZKkVatWnfDPA6YjUMBJqq2t1eTJk5WZmans7Gx9/vnnysvLa9bPZmdn6+OPP/ZeGf6/P4uoOXJzc/X0009r2LBhamxsPOGfB0zH1cwBAEZiDwoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkf4/w7b2b4+Y0dwAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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TpL07bff6uyzz/Zu4/8f4lu5cqXeeustSdLBgwf1n//8R71791bXrl01bNgwSVJCQoLCwsLUtWtXJSQk6Msvv7TxOwB0HAIFdKB58+bptttuU2pqqnbs2OGNkCSFh4f7/HzLspSVlaX77rvP57o7duzQe++9p1dffVXh4eGaNGmSmpubJUldu3aVw+GQdPyq3WFhYd6PPR7P6XxpgN9xmjnQgerr6+V0OiVJr7/+ervrDRkyRBs3bpQkvfvuuzpy5Igk6ZprrtGmTZt0+PBhSVJdXV27ezz19fWKjIxUeHi4PvvsM3344Ycd+aUAAcceFHCampqavIfRJGnKlCnKycnR9OnTFRkZqauvvlr79+8/4efm5ORo5syZeuONNzRo0CDFxMSoR48eio6O1owZM3T77berpaVFXbt2VW5uri644II22xg2bJheeeUVjRo1ShdddJEGDRpk29cKBAJXMwcCwO12q0uXLgoNDdXu3bv16KOPqqioKNBjAUZhDwoIgAMHDmjGjBnevaS5c+cGeiTAOOxBAQCMxEkSAAAjESgAgJEIFADASAQKAGAkAgUAMNL/An4CNng646N9AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "g = sns.catplot(x=\"SmallFam\", y=\"Survived\", data=train_df, height=6, \n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "g.set_xticklabels(['No', 'Yes'])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()\n",
        "g = sns.catplot(x=\"LargeFam\", y=\"Survived\", data=train_df, height=6, \n",
        "                   kind=\"bar\", palette=\"muted\")\n",
        "g.set_xticklabels(['No', 'Yes'])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 305,
      "id": "b5a5980c",
      "metadata": {
        "id": "b5a5980c"
      },
      "outputs": [],
      "source": [
        "# FARE \n",
        "\n",
        "def is_low_fare(x):\n",
        "    if float(x) <= 10:\n",
        "        return 1\n",
        "    else:\n",
        "        return 0\n",
        "\n",
        "train_df['LowFare'] = train_df.Fare.apply(is_low_fare)\n",
        "test_df['LowFare'] = test_df.Fare.apply(is_low_fare)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 306,
      "id": "46143489",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 441
        },
        "id": "46143489",
        "outputId": "e99b761f-6693-4f5a-8294-6845c8b43cb9"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x432 with 1 Axes>"
            ],
            "image/png": "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KACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwEoECABiJQAEAjESgAABGIlAAACMRKACAkQgUAMBIBAoAYCQCBQAwkttA/fnPf/bEHAAANODrboHZs2ertrZWqampGjFihIKCgjwxFwDAy7kN1IoVK3TkyBG9/fbbGj16tG688UaNHj1at99+uyfmAwB4KbeBkqSrr75aU6dOVVRUlObMmaP9+/fL6XRq2rRpSkhIsHpGAIAXchuoAwcOKD8/X1u3btVtt92ml156SX379pXD4dA999xDoAAAlnAbqDlz5igtLU3Tpk1T+/btXffb7XZNmTLF0uEAAN7L7Vl8Q4cO1ahRoxrE6dVXX5UkjRo1yrrJAABezW2g1qxZ0+i+d955x5JhAAC4oMldfOvXr9f69ev11Vdf6aGHHnLdX1lZqeDgYI8MBwDwXk0Gql+/fgoNDdXp06f1z//8z677O3bsqN69e3tkOACA92oyUFdddZWuuuoq/elPf/LkPAAASGomUBkZGXrjjTfUr18/2Ww21/1Op1M2m0179+71yIAAAO/UZKDeeOMNSdJHH33ksWEAALigyUCdOXOm2S8MCQlp8WEAALigyUCNHj1aNptNTqez0WM2m01FRUWWDgYA8G5NBmrLli2enAMAgAaaDNQXX3yhXr16ad++fRd9vG/fvpYNBQBAk4FatmyZnnrqKc2dO7fRYzabTcuXL7d0MACAd2syUE899ZQk6bXXXvPYMAAAXOD2auY1NTVasWKF9uzZI5vNpv79+ysjI0P+/v6emA8A4KXcXix2+vTpOnTokH71q1/p3nvv1eHDh/XYY495YjYAgBdzuwV16NAhbdiwwXV74MCBSkpKsnQoAADcbkHdcMMN+vjjj123P/nkE0VFRVk6FAAATW5BpaSkSJLOnTune+65Rz/96U8lSV9//bWuvfZaz0wHAPBaTQbqpZde8uQcAAA00Oyf2/i+U6dOqaamxvKBAACQLuEkiaKiIj377LM6ceKEunTpoq+//lq9evVSQUGBJ+YDAHgptydJzJ8/X3/605909dVXa8uWLVq2bJluuukmT8wGAPBibgPl6+urzp07q76+XvX19Ro4cKA+++wzT8wGAPBibnfxderUSZWVlRowYIAeffRRdenSRR06dPDEbAAAL+Z2C2rRokVq3769nnjiCd1xxx3q2bOnFi9e7InZAABezO0WVIcOHXTy5EmVlJQoODhYgwcPVufOnT0xGwDAi7ndglq5cqXS09P13nvvaePGjbr77ru1atUqT8wGAPBibregcnNz9c4777i2mk6fPq177rlHaWlplg8HAPBebregOnfurI4dO7pud+zY8ZJ38RUXFysxMVHx8fFasmRJo8ffeOMNpaSkaOTIkcrIyNDhw4cvY3QAwJWsyS2ovLw8SVLPnj01duxYxcXFyWazqaioSL1793a74rq6OmVnZysvL092u11paWmKjY3Vdddd51omJSVFGRkZks7/QnBOTo6WLl36Y18TAOAK0GSgKisrJZ0PVM+ePV33x8XFXdKKS0pKFBERofDwcElScnKyioqKGgQqMDDQ9Xl1dbVsNtvlTQ8AuGI1GajMzMwGty8E6/u7+5rjcDjUvXt312273a6SkpJGy/3xj39UXl6evvvuO7366qtu1xsY6C9fX59LmgFoDSEh/J4g0BLcniRx8OBBTZ8+XWVlZZLOH5N69tlnFRkZ2SID3Hvvvbr33nu1bt06LV68WM8++2yzy1dUtNwFa0NDg1psXcAFZ85UtfYIQJvS1PditydJZGVlacaMGXr//ff1/vvv6/HHH9esWbPcPqHdbldpaanrtsPhkN1ub3L55ORkbd682e16AQDewW2gqqqqNHDgQNftf/qnf1JVlfufEKOjo3XkyBEdO3ZMtbW1KigoUGxsbINljhw54vr8gw8+UERExGWMDgC4krndxRceHq4XX3xRI0eOlCStXbvWdeJDsyv29VVWVpYmTJiguro6jRkzRpGRkZo/f76ioqIUFxen119/XTt27JCvr686derkdvceAMB72JxOp7O5BcrKyrRgwQLt2bNHNptN/fv3V2ZmpoKDgz01YwMnT5a32LpCQ4P0y6wPWmx9wIrsIS36HgW8QVPHoJrdgqqrq1NmZqZee+01S4YCAKApzR6D8vHxUbt27VRezk+EAADPuqSrmaekpOi2225r8Hegfvvb31o6GADAu7kNVEJCghISEjwxCwAALm4DlZqaqtraWn355Zey2Wy65ppr5Ofn54nZAABezG2gtm7dqqysLPXs2VNOp1NfffWVZs+erZiYGE/MBwDwUm4DlZOTo+XLl7t+ifbo0aOaOHEigQIA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/ft/1ByxsbGuGVavXq24uLiLLvff//3fOnPmjM6ePavNmzfr5ptv1oABA7R582ZVV1erqqpKmzdv1oABA9S1a1edOnVKp0+fVm1trT744AO3c9xyyy2u3XgHDx7Un//8Z0nnd+84nU4lJiZq6tSpP/r1wnodO3ZUaGioduzYIen8SQnbtm1T//79m/26pt4DP//5z7V3717XMaCqqir95S9/+cHzDR48WNu3b1dZWZnKysq0fft2DR48+KLLFhYWSpI2bNjgOqbUr18/FRQUSJLWrVunAQMGSJKuuuoq7du3T9L5Py30/UBdzLXXXttdhqIAAAR5SURBVKvjx4/r6NGjkuRapyQdPXpUvXv31sSJExUdHf2jXm9bwhaUIWw2m1588UXNnj1bixYtUn19vWJiYjRt2jTXMomJiXr66ac1efJk132TJ0/WM888oxEjRqi+vl5hYWF6+eWX3T7ftGnTtHPnTp0+fVp33nmnHnnkEaWnp2vixImaOnWqVq1apZ/+9Kd64YUXLvr1N954ox555BE5HA6NGDFC0dHRks6fHZieni5JSktL0w033CBJevjhh5Weni673d5oa+9iMjIyNHPmTA0bNky9evVyHZ87ceKEZs6c6frJ9Pv/PjDXc889p9mzZ2vu3LmSzr8fevbs2ezXNPUe6NKli3JycjRt2jTXr2FMnTpV11xzTbPrKykpUWZmpr799lu9//77WrBggQoKChQSEqLJkycrLS3NNVtTu8bLysqUkpIiPz8/1xbcrFmzNHPmTC1dutQ1mySNHTtWkydP1ogRI3THHXeoQ4cOzc7n7++v7OxsTZw4UQEBAa7jTZL06quv6sMPP5TNZlNkZKTuvPPOZtd1peBSR7hs+fn5+uyzz5SVldXaowAeExsbq1WrVjX79+rQstjFBwAwEltQAAAjsQUFADASgQIAGIlAAQCMxGnmQAvq16+fPvrooxZZ14IFC/TWW2+5zhq744479Oijj7bIuoG2gEABBnvggQf061//+rK+5ty5c15zKRxc2djFB1js888/19ixY5WSkqKHH35YZWVlOnXqlOvipwcOHFDv3r1d15MbOnSoqqurL7qut956S2PGjNGIESP0yCOPuJabMWOGsrKylJ6ernnz5uno0aP69a9/rdGjR+uXv/ylvvjiC8+8WKAFESjAYtOnT9ejjz6qdevW6frrr9fChQvVtWtX1dTUqKKiQrt371ZUVJR2796t48ePq2vXrq7LWy1btsz1Zxa2bdum+Ph4vf3221q7dq2uvfZarVq1yvU8DodDb775pmbOnKlZs2Zp1qxZys/P1+OPP67Zs2e31ssHfjD2AwAWKi8vV3l5uevio6mpqZoyZYqk88er9uzZo127dumhhx7Stm3b5HQ6G1yj7h938e3cuVMvvPCCysvLVVlZ2eCacXfddZd8fHxUWVmpjz76yPU8kiz7y8yAlQgU0EoGDBigPXv26Ouvv1ZcXJxeeeUVSdKQIUOa/JoZM2Zo0aJF6tOnj/Lz8xtchfvCVpfT6VSnTp20Zs0aS+cHrMYuPsBCQUFB6tSpk+vvYq1Zs0a33HKLpPOBWrt2rSIiItSuXTsFBweruLi42at8V1ZWKjQ0VN99953WrVt30WUCAwMVFhbmuvK20+nUgQMHWviVAdZjCwpoQdXV1Q2uND1+/Hg9++yz+t3vfqfq6mqFh4e7rnYdFhYmp9PpClb//v1VWlqq4ODgJtc/ZcoUpaenq0uXLrrppptcV7v+R/PmzdOTTz6pxYsX69y5c0pKSnL9iRagreBafAAAI7GLDwBgJAIFADASgQIAGIlAAQCMRKAAAEYiUAAAIxEoAICR/j9+rNtG8jggGQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "g = sns.catplot(x=\"LowFare\", y=\"Survived\", data=train_df, height=6,kind=\"bar\", palette=\"muted\")\n",
        "g.despine(left=True)\n",
        "g.set_xticklabels(['Over 10 pounds', 'Under 10 pounds'])\n",
        "g.set_ylabels(\"survival probability\")\n",
        "plt.show()\n",
        "#Higher the fare better the passenger class and hence higher the probability of survival"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "fbb5ba8e",
      "metadata": {
        "id": "fbb5ba8e"
      },
      "source": [
        "# CLASSIFICATION MODEL"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 331,
      "id": "282deec6",
      "metadata": {
        "id": "282deec6"
      },
      "outputs": [],
      "source": [
        "# BUILDING THE MODEL\n",
        "\n",
        "from xgboost import XGBClassifier\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
        "\n",
        "# we don't need the PassengerIds anymore and we don't want them to influence the\n",
        "# classifier, we will need the PassengerIds for the training set later on when \n",
        "# we write the output file with all the the predictions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 332,
      "id": "217426de",
      "metadata": {
        "id": "217426de"
      },
      "outputs": [],
      "source": [
        "X = train_df.drop(['Survived'], axis=1)\n",
        "Y = train_df.Survived\n",
        "\n",
        "x_train, x_test, y_train, y_test = train_test_split(X, Y) # default split is 75% train and 25% test"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 333,
      "id": "35624dbf",
      "metadata": {
        "id": "35624dbf"
      },
      "outputs": [],
      "source": [
        "fixed_params = {'objective':'binary:logistic',\n",
        "                'scale_pos_weight':1.605\n",
        "                }\n",
        "\n",
        "xgb = XGBClassifier(**fixed_params)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 334,
      "id": "bf4b11c0",
      "metadata": {
        "id": "bf4b11c0"
      },
      "outputs": [],
      "source": [
        "test_params = {'n_estimators':np.array([25, 50, 100, 150, 200]),\n",
        "               'learning_rate':np.logspace(-4, -1, 4),\n",
        "               'max_depth':np.array([3, 4, 5, 6, 7]),\n",
        "               'gamma':np.array([0.0, 0.1]),\n",
        "               'max_delta_step':np.array([0.0, 0.001, 0.01]),\n",
        "               'reg_lambda':np.array([0.01, 0.1])\n",
        "               }"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 335,
      "id": "e89e0fdc",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 363
        },
        "id": "e89e0fdc",
        "outputId": "fec15a69-18d4-484b-9264-fab1008d1823"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fitting 3 folds for each of 1200 candidates, totalling 3600 fits\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-335-5b50f79bb94e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m grid_search = GridSearchCV(xgb, test_params, n_jobs=-1, verbose=1,\n\u001b[1;32m      3\u001b[0m                            scoring=my_score, cv=3)\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mgrid_search\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_search.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, groups, **fit_params)\u001b[0m\n\u001b[1;32m    889\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mresults\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    890\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 891\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_run_search\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mevaluate_candidates\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    892\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    893\u001b[0m             \u001b[0;31m# multimetric is determined here because in the case of a callable\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_search.py\u001b[0m in \u001b[0;36m_run_search\u001b[0;34m(self, evaluate_candidates)\u001b[0m\n\u001b[1;32m   1390\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_run_search\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevaluate_candidates\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1391\u001b[0m         \u001b[0;34m\"\"\"Search all candidates in param_grid\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1392\u001b[0;31m         \u001b[0mevaluate_candidates\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mParameterGrid\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparam_grid\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1393\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1394\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_search.py\u001b[0m in \u001b[0;36mevaluate_candidates\u001b[0;34m(candidate_params, cv, more_results)\u001b[0m\n\u001b[1;32m    849\u001b[0m                     )\n\u001b[1;32m    850\u001b[0m                     for (cand_idx, parameters), (split_idx, (train, test)) in product(\n\u001b[0;32m--> 851\u001b[0;31m                         \u001b[0menumerate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcandidate_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0menumerate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msplit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgroups\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    852\u001b[0m                     )\n\u001b[1;32m    853\u001b[0m                 )\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, iterable)\u001b[0m\n\u001b[1;32m   1054\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1055\u001b[0m             \u001b[0;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mretrieval_context\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1056\u001b[0;31m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mretrieve\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1057\u001b[0m             \u001b[0;31m# Make sure that we get a last message telling us we are done\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1058\u001b[0m             \u001b[0melapsed_time\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtime\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtime\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_start_time\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/joblib/parallel.py\u001b[0m in \u001b[0;36mretrieve\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    933\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    934\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'supports_timeout'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 935\u001b[0;31m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_output\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mextend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mjob\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    936\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    937\u001b[0m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_output\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mextend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mjob\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/joblib/_parallel_backends.py\u001b[0m in \u001b[0;36mwrap_future_result\u001b[0;34m(future, timeout)\u001b[0m\n\u001b[1;32m    540\u001b[0m         AsyncResults.get from multiprocessing.\"\"\"\n\u001b[1;32m    541\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 542\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfuture\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresult\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    543\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mCfTimeoutError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    544\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mTimeoutError\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.7/concurrent/futures/_base.py\u001b[0m in \u001b[0;36mresult\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    428\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__get_result\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    429\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 430\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_condition\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwait\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    431\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    432\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_state\u001b[0m \u001b[0;32min\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mCANCELLED\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mCANCELLED_AND_NOTIFIED\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.7/threading.py\u001b[0m in \u001b[0;36mwait\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    294\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m    \u001b[0;31m# restore state no matter what (e.g., KeyboardInterrupt)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    295\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mtimeout\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 296\u001b[0;31m                 \u001b[0mwaiter\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0macquire\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    297\u001b[0m                 \u001b[0mgotit\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    298\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ],
      "source": [
        "my_score = 'accuracy'\n",
        "grid_search = GridSearchCV(xgb, test_params, n_jobs=-1, verbose=1,\n",
        "                           scoring=my_score, cv=3)\n",
        "grid_search.fit(x_train, y_train)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "6cf2fa58",
      "metadata": {
        "id": "6cf2fa58"
      },
      "source": [
        "# MODEL EVALUATION"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "id": "5887ecdb",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 235
        },
        "id": "5887ecdb",
        "outputId": "946d1031-39e6-4483-8396-696761f39e9c"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-a5821ba162ac>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mprint\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;34m'Best %s score: %0.3f'\u001b[0m\u001b[0;34m%\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmy_score\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgrid_search\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbest_score_\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mprint\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;34m'Best Parameters:'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mbest_parameters\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgrid_search\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbest_estimator_\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_params\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mparam_name\u001b[0m \u001b[0;32min\u001b[0m \u001b[0msorted\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest_params\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mkeys\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0mprint\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;34m'\\t%s: %r'\u001b[0m\u001b[0;34m%\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mparam_name\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbest_parameters\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mparam_name\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'my_score' is not defined"
          ]
        }
      ],
      "source": [
        "print ('Best %s score: %0.3f'%(my_score, grid_search.best_score_))\n",
        "print ('Best Parameters:')\n",
        "best_parameters = grid_search.best_estimator_.get_params()\n",
        "for param_name in sorted(test_params.keys()):\n",
        "    print ('\\t%s: %r'%(param_name, best_parameters[param_name]))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "2047e3d9",
      "metadata": {
        "id": "2047e3d9"
      },
      "outputs": [],
      "source": [
        "predictions_test = grid_search.predict(x_test)\n",
        "\n",
        "print ('Scores for final validation set:')\n",
        "print ('\\taccuracy score: %f'%accuracy_score(y_test, predictions_test))\n",
        "print ('\\tprecision score: %f'%precision_score(y_test, predictions_test))\n",
        "print ('\\trecall score: %f'%recall_score(y_test, predictions_test))\n",
        "print ('\\tf1 score: %f'%f1_score(y_test, predictions_test))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "1aefa509",
      "metadata": {
        "id": "1aefa509"
      },
      "outputs": [],
      "source": [
        ""
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "t-BEdfcHN80W"
      },
      "id": "t-BEdfcHN80W",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Random forest"
      ],
      "metadata": {
        "id": "Sz5lNPj6N-0v"
      },
      "id": "Sz5lNPj6N-0v"
    },
    {
      "cell_type": "code",
      "source": [
        "# Algorithms\n",
        "from sklearn import linear_model\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.linear_model import Perceptron\n",
        "from sklearn.linear_model import SGDClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.svm import SVC, LinearSVC\n",
        "from sklearn.naive_bayes import GaussianNB"
      ],
      "metadata": {
        "id": "1YOOMrmxQBak"
      },
      "id": "1YOOMrmxQBak",
      "execution_count": 336,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = train_df.drop(\"Survived\", axis=1)\n",
        "Y_train = train_df[\"Survived\"]\n",
        "X_test  = test_df.copy()"
      ],
      "metadata": {
        "id": "SSWtL7k1OF7V"
      },
      "id": "SSWtL7k1OF7V",
      "execution_count": 337,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "random_forest = RandomForestClassifier(n_estimators=100)\n",
        "random_forest.fit(X_train, Y_train)\n",
        "\n",
        "Y_pred = random_forest.predict(X_test)\n",
        "\n",
        "random_forest.score(X_train, Y_train)\n",
        "acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)"
      ],
      "metadata": {
        "id": "XDMT_NvCOFy7"
      },
      "id": "XDMT_NvCOFy7",
      "execution_count": 338,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#KNN\n",
        "knn = KNeighborsClassifier(n_neighbors = 3) \n",
        "knn.fit(X_train, Y_train)  \n",
        "Y_pred = knn.predict(X_test)  \n",
        "acc_knn = round(knn.score(X_train, Y_train) * 100, 2)"
      ],
      "metadata": {
        "id": "oT8qtdjTQETi"
      },
      "id": "oT8qtdjTQETi",
      "execution_count": 339,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "linear_svc = LinearSVC()\n",
        "linear_svc.fit(X_train, Y_train)\n",
        "\n",
        "Y_pred = linear_svc.predict(X_test)\n",
        "\n",
        "acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NaZ9dtKaQqw8",
        "outputId": "7e803205-f10b-4408-f418-66491c119451"
      },
      "id": "NaZ9dtKaQqw8",
      "execution_count": 340,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/svm/_base.py:1208: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.\n",
            "  ConvergenceWarning,\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "logreg = LogisticRegression()\n",
        "logreg.fit(X_train, Y_train)\n",
        "\n",
        "Y_pred = logreg.predict(X_test)\n",
        "\n",
        "acc_log = round(logreg.score(X_train, Y_train) * 100, 2)\n"
      ],
      "metadata": {
        "id": "b_cpqGFRQssS"
      },
      "id": "b_cpqGFRQssS",
      "execution_count": 341,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sgd = linear_model.SGDClassifier(max_iter=5, tol=None)\n",
        "sgd.fit(X_train, Y_train)\n",
        "Y_pred = sgd.predict(X_test)\n",
        "\n",
        "sgd.score(X_train, Y_train)\n",
        "\n",
        "acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)"
      ],
      "metadata": {
        "id": "u3Qk6gy8SgLO"
      },
      "id": "u3Qk6gy8SgLO",
      "execution_count": 342,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "gaussian = GaussianNB() \n",
        "gaussian.fit(X_train, Y_train)  \n",
        "Y_pred = gaussian.predict(X_test)  \n",
        "acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)"
      ],
      "metadata": {
        "id": "5kW59wyYSf-I"
      },
      "id": "5kW59wyYSf-I",
      "execution_count": 343,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "perceptron = Perceptron(max_iter=5)\n",
        "perceptron.fit(X_train, Y_train)\n",
        "\n",
        "Y_pred = perceptron.predict(X_test)\n",
        "\n",
        "acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JoMnObBjSfvN",
        "outputId": "cc07a777-9173-46bd-93cf-485322fcf42f"
      },
      "id": "JoMnObBjSfvN",
      "execution_count": 344,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_stochastic_gradient.py:700: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.\n",
            "  ConvergenceWarning,\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "decision_tree = DecisionTreeClassifier()\n",
        "decision_tree.fit(X_train, Y_train)  \n",
        "Y_pred = decision_tree.predict(X_test)  \n",
        "acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)"
      ],
      "metadata": {
        "id": "8FhZ65S5Swrs"
      },
      "id": "8FhZ65S5Swrs",
      "execution_count": 345,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "results = pd.DataFrame({\n",
        "    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', \n",
        "              'Random Forest', 'Naive Bayes', 'Perceptron', \n",
        "              'Stochastic Gradient Decent', \n",
        "              'Decision Tree'],\n",
        "    'Score': [acc_linear_svc, acc_knn, acc_log, \n",
        "              acc_random_forest, acc_gaussian, acc_perceptron, \n",
        "              acc_sgd, acc_decision_tree]})\n",
        "result_df = results.sort_values(by='Score', ascending=False)\n",
        "result_df = result_df.set_index('Score')\n",
        "result_df.head(9)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 331
        },
        "id": "3xbdnIzSQsgr",
        "outputId": "7e865a3a-a674-47c2-f7c1-75cc8729dcf9"
      },
      "id": "3xbdnIzSQsgr",
      "execution_count": 346,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                            Model\n",
              "Score                            \n",
              "90.57               Random Forest\n",
              "90.57               Decision Tree\n",
              "87.43                         KNN\n",
              "81.14         Logistic Regression\n",
              "80.92     Support Vector Machines\n",
              "77.78                  Perceptron\n",
              "75.08  Stochastic Gradient Decent\n",
              "72.62                 Naive Bayes"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-bd5407b8-2e5f-463d-814a-b52d9a305577\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Model</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Score</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>90.57</th>\n",
              "      <td>Random Forest</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>90.57</th>\n",
              "      <td>Decision Tree</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>87.43</th>\n",
              "      <td>KNN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>81.14</th>\n",
              "      <td>Logistic Regression</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>80.92</th>\n",
              "      <td>Support Vector Machines</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>77.78</th>\n",
              "      <td>Perceptron</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75.08</th>\n",
              "      <td>Stochastic Gradient Decent</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>72.62</th>\n",
              "      <td>Naive Bayes</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-bd5407b8-2e5f-463d-814a-b52d9a305577')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-bd5407b8-2e5f-463d-814a-b52d9a305577 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-bd5407b8-2e5f-463d-814a-b52d9a305577');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 346
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# K folds cross validation"
      ],
      "metadata": {
        "id": "kJ7fpS8jS9zE"
      },
      "id": "kJ7fpS8jS9zE"
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "rf = RandomForestClassifier(n_estimators=100)\n",
        "scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = \"accuracy\")\n",
        "print(\"Scores:\", scores)\n",
        "print(\"Mean:\", scores.mean())\n",
        "print(\"Standard Deviation:\", scores.std())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w8YRdhfsTCR5",
        "outputId": "f549c2c4-8e18-447f-8dd5-af15a271003b"
      },
      "id": "w8YRdhfsTCR5",
      "execution_count": 347,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Scores: [0.73333333 0.82022472 0.74157303 0.80898876 0.86516854 0.83146067\n",
            " 0.82022472 0.78651685 0.83146067 0.78651685]\n",
            "Mean: 0.8025468164794007\n",
            "Standard Deviation: 0.03909251272011316\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})\n",
        "importances = importances.sort_values('importance',ascending=False).set_index('feature')\n",
        "importances.head(15)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 488
        },
        "id": "8FszlUfWTMVi",
        "outputId": "e5407b1d-9432-464e-dbb4-bd5b970c9bd6"
      },
      "id": "8FszlUfWTMVi",
      "execution_count": 348,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "          importance\n",
              "feature             \n",
              "Sex            0.335\n",
              "Age            0.168\n",
              "Fare           0.109\n",
              "Pclass         0.101\n",
              "Embarked       0.060\n",
              "LowFare        0.049\n",
              "FamSize        0.047\n",
              "Parch          0.031\n",
              "SibSp          0.029\n",
              "SmallFam       0.027\n",
              "LargeFam       0.018\n",
              "Child          0.017\n",
              "Alone          0.009"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-bbaf919a-1ac2-4eb3-9b6a-67216e811f5c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>importance</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>feature</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Sex</th>\n",
              "      <td>0.335</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Age</th>\n",
              "      <td>0.168</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Fare</th>\n",
              "      <td>0.109</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Pclass</th>\n",
              "      <td>0.101</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Embarked</th>\n",
              "      <td>0.060</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>LowFare</th>\n",
              "      <td>0.049</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>FamSize</th>\n",
              "      <td>0.047</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Parch</th>\n",
              "      <td>0.031</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>SibSp</th>\n",
              "      <td>0.029</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>SmallFam</th>\n",
              "      <td>0.027</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>LargeFam</th>\n",
              "      <td>0.018</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Child</th>\n",
              "      <td>0.017</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Alone</th>\n",
              "      <td>0.009</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-bbaf919a-1ac2-4eb3-9b6a-67216e811f5c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-bbaf919a-1ac2-4eb3-9b6a-67216e811f5c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-bbaf919a-1ac2-4eb3-9b6a-67216e811f5c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 348
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "importances.plot.bar()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 341
        },
        "id": "etPLAoNsTTpG",
        "outputId": "56e1baf0-638d-46db-e38f-ae9c6513961b"
      },
      "id": "etPLAoNsTTpG",
      "execution_count": 349,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fe1ec5f2f90>"
            ]
          },
          "metadata": {},
          "execution_count": 349
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_df  = train_df.drop(\"Child\", axis=1)\n",
        "test_df  = test_df.drop(\"Child\", axis=1)\n",
        "\n",
        "train_df  = train_df.drop(\"Alone\", axis=1)\n",
        "test_df  = test_df.drop(\"Alone\", axis=1)"
      ],
      "metadata": {
        "id": "eg2kuRjQTYYa"
      },
      "id": "eg2kuRjQTYYa",
      "execution_count": 350,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Random Forest\n",
        "\n",
        "random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)\n",
        "random_forest.fit(X_train, Y_train)\n",
        "Y_prediction = random_forest.predict(X_test)\n",
        "\n",
        "random_forest.score(X_train, Y_train)\n",
        "\n",
        "acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)\n",
        "print(round(acc_random_forest,2,), \"%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vj_DVHB9To8W",
        "outputId": "2f8e1e3d-8ead-411b-a59e-4ed8348d9552"
      },
      "id": "vj_DVHB9To8W",
      "execution_count": 351,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "90.57 %\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"oob score:\", round(random_forest.oob_score_, 4)*100, \"%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KUhCKMCKT5Dh",
        "outputId": "2f31b361-d6e8-4efb-f1df-113c7ad17fff"
      },
      "id": "KUhCKMCKT5Dh",
      "execution_count": 352,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "oob score: 81.37 %\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import precision_score, recall_score\n",
        "from sklearn.model_selection import cross_val_predict\n",
        "from sklearn.metrics import confusion_matrix\n",
        "predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)\n",
        "print(confusion_matrix(Y_train, predictions))\n",
        "\n",
        "print(\"Precision:\", precision_score(Y_train, predictions))\n",
        "print(\"Recall:\",recall_score(Y_train, predictions))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eV6MoomCULBb",
        "outputId": "575bcb7e-5cd6-4518-863a-74318f9e9566"
      },
      "id": "eV6MoomCULBb",
      "execution_count": 355,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[469  80]\n",
            " [102 240]]\n",
            "Precision: 0.75\n",
            "Recall: 0.7017543859649122\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "TITANIC_PROJECT.ipynb",
      "provenance": [],
      "collapsed_sections": [
        "b8df3939",
        "6cf2fa58"
      ]
    },
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
      "version": "3.9.7"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}