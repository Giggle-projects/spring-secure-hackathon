//package se.ton.t210.configuration.filter;
//
//import se.ton.t210.configuration.http.XssCleanHttpRequestWrapper;
//
//import javax.servlet.*;
//import javax.servlet.annotation.WebFilter;
//import javax.servlet.http.HttpServletRequest;
//import java.io.IOException;
//
////@WebFilter(urlPatterns = {"/api/*"})
//public class XssReplacingFilter implements Filter  {
//
//    @Override
//    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
//        chain.doFilter(new XssCleanHttpRequestWrapper((HttpServletRequest) request), response);
//    }
//}
