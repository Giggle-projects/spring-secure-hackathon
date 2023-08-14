## Access log

Access 정보를 관리한다.    
사용자 측 IP와 사용한 토큰 정보를 로깅하여 해킹에 대비한다.    

IP로 사용자 요청을 막을 수 있다.    

### Metrics
```
1. Request url
2. Request status
3. Remote address     (User IP)
4. Remote header      (User token)
5. Response status
6. Response message 
```

### Format
```
[HTTP_REQ] server : GET /api/score, remote : 0:0:0:0:0:0:0:1 none
[HTTP_RES] 500 INTERNAL_SERVER_ERROR
```

### File logging strategy

콘솔로 출력만 하는 것이 아니라 파일로 이를 관리한다.
`/logs`를 baseDirec로 해서 로그 파일이 생성된다.    

```
현재 로그는 'server-access-log.log'로 파일에 저장된다.
하루의 로그가 압축되어 'YYYY-MM-DD.log.zip'로 날짜에 해당하는 로그가 압축되어 저장된다.
15일이 지난 로그 파일은 삭제된다.
```

### Log4j2
이런 로그 정책과 포맷, 설정은 Log4j2 라이브러리를 사용하여 자동화되고 관리되고 있다.  

### Enable property

요청과 응답을 나눠 true, false로 엑세스 로깅 사용 여부를 정할 수 있다.

```
global.log.http.access.request.enable=true
global.log.http.access.response.enable=false
```
