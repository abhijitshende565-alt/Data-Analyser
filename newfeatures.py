import streamlit as st


def search_by_word(df):
    st.subheader("Search in Data")

    word = st.text_input("Enter word to search")

    if word:
        df_str = df.astype(str)

        filtered_df = df[
            df_str.apply(
                lambda row: row.str.contains(word, case=False).any(),
                axis=1
            )
        ]

        if not filtered_df.empty:
            st.success(f"{len(filtered_df)} matching rows found")
            st.dataframe(filtered_df)
        else:
            st.warning("No matching data found")






