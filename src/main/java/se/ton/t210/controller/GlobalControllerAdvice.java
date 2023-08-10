package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import se.ton.t210.exception.AuthException;

@ControllerAdvice
public class GlobalControllerAdvice {

    @ExceptionHandler(AuthException.class)
    public ResponseEntity<String> handledMemberException(AuthException e) {
        return ResponseEntity.status(e.getHttpStatus()).body(e.getMessage());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<String> handledException(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(e.getMessage());
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<String> unhandledException(Exception e) {
        e.printStackTrace();
        return ResponseEntity.internalServerError().body("Unhandled exception");
    }
}
