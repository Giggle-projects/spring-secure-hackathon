package se.ton.t210.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import se.ton.t210.exception.AuthException;

import javax.validation.ConstraintViolationException;

@ControllerAdvice
public class GlobalControllerAdvice {

    private static final Logger LOGGER = LoggerFactory.getLogger(GlobalControllerAdvice.class);

    @ExceptionHandler(AuthException.class)
    public ResponseEntity<String> handledMemberException(AuthException e) {
        return ResponseEntity.status(e.getHttpStatus()).body(e.getMessage());
    }

    @ExceptionHandler({ConstraintViolationException.class, MethodArgumentTypeMismatchException.class})
    public ResponseEntity<String> invalidInputPrams() {
        return ResponseEntity.badRequest().body("Invalid input request");
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<String> handledException(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(e.getMessage());
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<String> unhandledException(Exception e) {
        LOGGER.error(e.getMessage());
        e.printStackTrace();
        return ResponseEntity.internalServerError().body("Unhandled exception");
    }
}
