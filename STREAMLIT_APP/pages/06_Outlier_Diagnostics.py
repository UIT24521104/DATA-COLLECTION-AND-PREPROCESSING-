"""
Streamlit page for comprehensive outlier analysis and diagnosis.

Displays:
1. Univariate outliers by feature (countries identified)
2. Multivariate outliers (country interactions)
3. Summary tables and visualizations
4. Export functionality
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from services.outlier_diagnostic_service import OutlierDiagnostics
from services.data_loader import load_latest_datasets
from utils.ui import set_page_config, load_css


ROOT_DIR = Path(__file__).resolve().parents[2]


def plot_outlier_summary(summary_df: pd.DataFrame) -> go.Figure:
    """Create visualization of outlier summary by country."""
    if summary_df.empty:
        return None
    
    # Sort by total outlier count
    summary_df['Total_Outliers'] = summary_df['Univariate_Count'] + summary_df['Multivariate_Count']
    summary_df_sorted = summary_df.sort_values('Total_Outliers', ascending=True).tail(20)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=summary_df_sorted['Country'],
        x=summary_df_sorted['Univariate_Count'],
        name='Univariate Outliers',
        orientation='h',
        marker=dict(color='#FF6B6B'),
    ))
    
    fig.add_trace(go.Bar(
        y=summary_df_sorted['Country'],
        x=summary_df_sorted['Multivariate_Count'],
        name='Multivariate Outliers',
        orientation='h',
        marker=dict(color='#4ECDC4'),
    ))
    
    fig.update_layout(
        title='🔍 Countries by Outlier Count (Top 20)',
        xaxis_title='Number of Outliers',
        yaxis_title='Country',
        barmode='stack',
        height=600,
        template='plotly_white',
        hovermode='closest',
    )
    
    return fig


def plot_anomaly_scores(multivariate_results: dict) -> go.Figure | None:
    """Create visualization of multivariate anomaly scores."""
    if not multivariate_results.get('outlier_countries'):
        return None
    
    data_points = []
    for country, info in multivariate_results['outlier_countries'].items():
        for score, year in zip(info['anomaly_scores'], info['years']):
            data_points.append({
                'Country': country,
                'Year': year,
                'Anomaly_Score': score,
            })
    
    df = pd.DataFrame(data_points)
    
    # Normalize anomaly scores to positive range for marker size (1-15)
    min_score = df['Anomaly_Score'].min()
    max_score = df['Anomaly_Score'].max()
    if min_score < max_score:
        df['Size'] = 1 + (df['Anomaly_Score'] - min_score) / (max_score - min_score) * 14
    else:
        df['Size'] = 5  # Default if all scores are the same
    
    fig = px.scatter(
        df,
        x='Year',
        y='Anomaly_Score',
        color='Country',
        size='Size',
        hover_data=['Country', 'Year', 'Anomaly_Score'],
        title='📊 Multivariate Outlier Scores (Isolation Forest)',
        template='plotly_white',
    )
    
    fig.update_layout(height=500)
    return fig


def display_univariate_outliers(univariate_results: dict):
    """Display univariate outliers organized by feature."""
    st.subheader('📌 Univariate Outliers by Feature')
    st.markdown("""
    **Definition**: Values that deviate significantly from the distribution of a single feature.
    **Method**: IQR (Interquartile Range) with 1.5× threshold
    """)
    
    # Create feature list
    features_with_outliers = [
        f for f, info in univariate_results.items()
        if info['outlier_countries']
    ]
    
    if not features_with_outliers:
        st.info('✅ No univariate outliers detected')
        return
    
    for feature in sorted(features_with_outliers):
        feature_info = univariate_results[feature]
        outlier_countries = feature_info['outlier_countries']
        
        # Feature header with statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Outliers', feature_info['total_outliers'])
        with col2:
            st.metric('% of Data', f"{feature_info['outlier_pct']:.1f}%")
        with col3:
            st.metric('Countries', len(outlier_countries))
        with col4:
            bounds = feature_info['statistics']
            st.metric('IQR', f"{bounds['IQR']:.2f}")
        
        # Create detailed table for this feature
        outlier_list = []
        for country in sorted(outlier_countries.keys()):
            country_info = outlier_countries[country]
            outlier_list.append({
                '🌍 Country': country,
                '📍 Count': country_info['count'],
                '📅 Years': ', '.join(map(str, country_info['years'])),
                '📊 Values': ', '.join([f"{v:.2f}" for v in country_info['values']]),
                '⚠️ Severity': country_info['severity'].upper(),
            })
        
        outlier_df = pd.DataFrame(outlier_list)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        # Show bounds
        with st.expander(f'📐 {feature} - Statistical Bounds'):
            bounds = feature_info['statistics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Q1**: {bounds['Q1']:.2f}")
            with col2:
                st.write(f"**Q3**: {bounds['Q3']:.2f}")
            with col3:
                st.write(f"**Lower**: {bounds['lower_bound']:.2f}")
            with col4:
                st.write(f"**Upper**: {bounds['upper_bound']:.2f}")
        
        st.divider()


def display_multivariate_outliers(multivariate_results: dict):
    """Display multivariate outliers (interaction-based)."""
    st.subheader('🔗 Multivariate Outliers (Feature Interactions)')
    st.markdown("""
    **Definition**: Countries that are unusual in the *combination* of multiple features.
    **Method**: Isolation Forest - detects anomalies based on feature interactions
    """)
    
    if not multivariate_results.get('outlier_countries'):
        st.info('✅ No significant multivariate outliers detected')
        return
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Outlier Points', multivariate_results['total_multivar_outliers'])
    with col2:
        st.metric('% of Data', f"{multivariate_results['outlier_pct']:.1f}%")
    with col3:
        st.metric('Affected Countries', len(multivariate_results['outlier_countries']))
    
    # Create detailed table
    multivar_list = []
    for country in sorted(multivariate_results['outlier_countries'].keys()):
        country_info = multivariate_results['outlier_countries'][country]
        multivar_list.append({
            '🌍 Country': country,
            '📍 Outlier Points': country_info['count'],
            '📅 Years': ', '.join(map(str, country_info['years'])),
            '🎯 Avg Score': f"{country_info['avg_score']:.3f}",
            '⚡ Max Score': f"{country_info['max_score']:.3f}",
        })
    
    if multivar_list:
        multivar_df = pd.DataFrame(multivar_list)
        st.dataframe(multivar_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = plot_anomaly_scores(multivariate_results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    set_page_config()
    load_css(ROOT_DIR / 'streamlit_app' / 'assets' / 'styles.css')
    
    st.title('🔍 Outlier Diagnostics - Signal vs Noise Analysis')
    
    st.markdown("""
    This diagnostic tool helps you understand which countries have outlier values and whether
    these represent **meaningful signals** (real insights) or **noise** (measurement errors).
    
    ### Analysis Types
    1. **Univariate Outliers**: Countries with extreme values in individual features
    2. **Multivariate Outliers**: Countries with unusual combinations across features
    """)
    
    # Load data
    with st.spinner('Loading data...'):
        try:
            datasets = load_latest_datasets(ROOT_DIR)
            # Use final processed data for more reliable outlier detection
            data = datasets.get('final')
            if data is None:
                data = datasets.get('merged')
        except Exception as e:
            st.error(f'❌ Error loading datasets: {e}')
            return
    
    if data is None or data.empty:
        st.error('❌ No data available')
        return
    
    st.divider()
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        iqr_threshold = st.slider(
            '🎯 IQR Threshold (univariate)',
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.5,
            help='Higher = more extreme outliers only'
        )
    
    with col2:
        contamination = st.slider(
            '📊 Expected Contamination %',
            min_value=1,
            max_value=20,
            value=10,
            help='% of data expected to be outliers'
        )
    
    with col3:
        run_analysis = st.button('🚀 Run Outlier Analysis', use_container_width=True)
    
    st.divider()
    
    if run_analysis or st.session_state.get('outlier_report_ready'):
        with st.spinner('🔬 Analyzing outliers...'):
            # Initialize diagnostics
            diagnostics = OutlierDiagnostics(data, country_col='Country', year_col='Year')
            
            # Generate report
            report = diagnostics.get_outlier_report(iqr_threshold=iqr_threshold)
            st.session_state['outlier_report_ready'] = True
            st.session_state['outlier_report'] = report
        
        # Display results
        tab1, tab2, tab3, tab4 = st.tabs([
            '📊 Summary',
            '📌 Univariate',
            '🔗 Multivariate',
            '💾 Export'
        ])
        
        with tab1:
            st.subheader('📊 Outlier Summary by Country')
            summary_df = report['summary']
            
            # Color code by risk level
            def risk_color(val):
                if val == 'High':
                    return 'background-color: #ff4444'
                elif val == 'Medium':
                    return 'background-color: #ffaa44'
                else:
                    return 'background-color: #44ff44'
            
            st.dataframe(
                summary_df.style.applymap(
                    risk_color,
                    subset=['Risk_Level']
                ),
                use_container_width=True,
                hide_index=True,
            )
            
            # Visualization
            fig = plot_outlier_summary(summary_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader('📈 Statistics')
            high_risk = len(summary_df[summary_df['Risk_Level'] == 'High'])
            medium_risk = len(summary_df[summary_df['Risk_Level'] == 'Medium'])
            low_risk = len(summary_df[summary_df['Risk_Level'] == 'Low'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('🔴 High Risk Countries', high_risk)
            with col2:
                st.metric('🟡 Medium Risk Countries', medium_risk)
            with col3:
                st.metric('🟢 Low Risk Countries', low_risk)
        
        with tab2:
            display_univariate_outliers(report['univariate_iqr'])
        
        with tab3:
            display_multivariate_outliers(report['multivariate'])
        
        with tab4:
            st.subheader('💾 Export Outlier Reports')
            st.markdown("""
            Download detailed outlier analysis for further investigation or reporting.
            """)
            
            if st.button('📥 Generate Export Files'):
                try:
                    output_dir = ROOT_DIR / 'pipeline_execution' / 'outputs' / 'latest'
                    files = diagnostics.export_outlier_report(output_dir, report)
                    
                    st.success('✅ Files generated successfully!')
                    
                    for file_type, file_path in files.items():
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label=f'📊 Download {file_type.replace("_", " ").title()}',
                                data=f.read(),
                                file_name=file_path.name,
                                mime='text/csv',
                            )
                
                except Exception as e:
                    st.error(f'❌ Error generating export files: {e}')


if __name__ == '__main__':
    main()
