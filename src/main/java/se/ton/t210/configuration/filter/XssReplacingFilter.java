package se.ton.t210.configuration.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.ton.t210.configuration.http.XssCleanHttpRequestWrapper;

import javax.servlet.*;
import javax.servlet.annotation.WebFilter;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;

@WebFilter(urlPatterns = {"*"})
public class XssReplacingFilter implements Filter  {

    private static final Logger LOGGER = LoggerFactory.getLogger(XssReplacingFilter.class);

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        LOGGER.info("PageScriptReplacingFilter");
        chain.doFilter(new XssCleanHttpRequestWrapper((HttpServletRequest) request), response);
    }
}
