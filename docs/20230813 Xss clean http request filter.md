## Xss clean filter 

사용자의 입력에서 스크립트로 동작할 수 있는 입력을 서버에서 filter로 선처리한다.     
특히 사용자 정보의 경우에는 이런 스크립트가 될 수 있는 문자를 직접 저장했다가 페이지에 노출되게 되었을 때 의도치 않은 동작으로 이어질 수 있다.    

### 예시
예를 들어 사용자가 본인 이름으로 아래 스크립트 자체를 넣었다고 가정하자. 
```
<script> alert('hack-script'); </script> 
```
서버에선 DB에 사용자 이름으로 이를 저장할거고 응답으로 이를 반환했을 때 페이지에서 이를 그대로 노출한다면 이는 공격 스크립트가 될 수 있는 것이다.

### XSS 실습

실제 프로젝트에서 이를 실습한다. 사용자 이름으로 하이퍼링크를 넘기고 페이지에선 이를 다른 처리없이 출력하니 사용자 이름인 척하는 외부 링크가 된 걸 볼 수 있다.    

### Filter와 HttpRequest / Naver Lucy filter ?

Naver에서 만든 이런 상황을 대비한 Xss filter 라이브러리가 있다. Lucy filter.    
근데 이 라이브러리는 헤더와 파라미터를 필터링하지만 내가 원했던 리퀘스트 바디는 커버하지 않았다.    

서버 요청을 처리하는 HttpServletRequest를 감싸는 wrapper를 정의하는 것으로 요청 정보를 커스텀 할 수 있도록 한다.    
그리고 설정한 요청 경로의 request에 이 wrapper를 전달하는 Filter를 정의하여 원하는 api에 xss를 막는 로직을 담을 수 있도록 하였다.    

모든 요청에 적용하려 하면 특수문자가 그래도 필요한 상황, 특히 동작을 명확하게 알지 못하는 외부 라이브러리를 사용하는데 문제가 있을 수 있다.   
이번의 경우에는 h2의 인증에서 이 필터가 추가되니 문제가 있었다.    

특수문자를 어떻게 처리할지가 명확한 api, Xss 공격과 프론트 노출 시 문제가 될 api 만 특정하여 적용한다. 

### Sample

1. 사용자 이름으로 스크립트를 숨겨 서버에 요청했다.
<img width="987" alt="Screenshot 2023-08-14 at 5 48 16 PM" src="https://github.com/Giggle-projects/spring-secure-hackathon/assets/46060746/162fcc10-a61a-40da-9634-42fca4d647df">

<br/>


2. 필터가 없는 서버는 이를 그대로 DB에 저장하고 반환한다. 페이지에 직접 노출되었다.
<img width="1294" alt="Screenshot 2023-08-14 at 5 44 42 PM" src="https://github.com/Giggle-projects/spring-secure-hackathon/assets/46060746/9fb3f8dc-83fb-473e-97ef-3444ed26e5dc">


<br/>

3. 필터를 적용 후 공격 스크립트를 숨길 수 있는 문자들을 치환하여 서버는 요청을 다루게 된다.
<img width="1352" alt="Screenshot 2023-08-14 at 6 37 17 PM" src="https://github.com/Giggle-projects/spring-secure-hackathon/assets/46060746/19d48275-2e9e-4ad8-97d4-4ca342694118">

