package se.ton.t210.utils.log;

import lombok.Getter;

import javax.servlet.http.HttpServletRequest;

@Getter
public class RequestLog {

    private static final String LOG_FORMAT = "[HTTP_REQ] server : %s %s, remote : %s %s";

    private final String remoteIp;
    private final String accessToken;
    private final String uri;
    private final String method;

    public RequestLog(HttpServletRequest request, String accessTokenHeaderKey) {
        this.remoteIp = remoteIp(request);
        this.accessToken = accessToken(request, accessTokenHeaderKey);
        this.uri = uri(request);
        this.method = method(request);
    }

    private static String remoteIp(HttpServletRequest request) {
        final String ip = request.getHeader("X-FORWARDED-FOR");
        if (ip == null) {
            return request.getRemoteAddr();
        }
        return ip;
    }

    private static String accessToken(HttpServletRequest request, String accessTokenHeaderKey) {
        return request.getHeader(accessTokenHeaderKey);
    }

    private static String uri(HttpServletRequest request) {
        return request.getRequestURI();
    }

    private static String method(HttpServletRequest request) {
        return request.getMethod();
    }

    public String asLog() {
        return String.format(LOG_FORMAT, method, uri, remoteIp, accessToken == null ? "none" : accessToken);
    }
}
