package se.ton.t210.exception;

/*
 Do not reveal this exception message to outer.
 This error is only for logging like issue on db, security logics, etc.
 */
public class InnerServiceException extends IllegalArgumentException {

  public InnerServiceException(String s) {
    super(s);
  }
}
