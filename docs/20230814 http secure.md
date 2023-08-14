## Http secure 

Http only 태그로도 Http 패킷 자체를 탈취당하면 cookie 정보를 확인할 수 있다.       
Https 통신 키를 탈취당하거나 개발자의 부주의로 Http로 통신되는 경우 암호화하지 않은 요청 정보로 cookie의 정보를 그대로 확인할 수 있다.      

서버 쪽에서 쿠키를 발급할 때 Http secure 태그를 함께 하는 것으로 브라우저가 HTTPS 통신 외에서는 쿠키를 전송하지 않도록 하였다.   

`https://nsinc.tistory.com/121`
