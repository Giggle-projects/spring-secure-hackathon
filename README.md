## Spring secure hackathon

### How to run [dev]

#### env container up
```
docker-compose -f docker-compose-dev.yaml up -d
```

#### env container down
```
docker-compose -f docker-compose-dev.yaml down
```

```
./gradlew build
java -jar build/libs/t210-0.0.1.jar 
```

### [BE - DevOPs]
- [x] Access token, Refresh token
- [] TLS setting
- [x] CRUD
- [x] email sender
- [x] CI/CD
- [] ip black list filter
- [x] ip - token logging
- [] personal information encrypt
- [] combine with FE
- [] block invalid FE request
- [x] xss filter
- [] block reusing - password
- [] response message login with old password
- [] BE deployment
- [] FE connection
- [] FE deployment
- [] Java secure coding guide double checking
- [] Stress test
- [] Api gateway traffic blocking
- [] Make document
- [] Vault configuration
- [] HA with AWS resources

### Notes
0. Figma
`https://www.figma.com/file/GlundLndn8GygquN1W2Gds/Untitled?type=design&node-id=1%3A1369&mode=design&t=IJStLHe23oxLCx8b-1`

1. Services
- 1-1 ApplicationType : PoliceOfficerMale, PoliceOfficerFemale, FireOfficerMale, FireOfficerFemale, CorrectionalOfficerMale, CorrectionalOfficerFemale
- 1-2 직렬은 회원 당 하나
- 1-3 일단 평균 -> int
- 1-4 직렬 당 시험 종목 목록은 5개
- 1-5 티어는 퍼센트지로 (Bronze, Silver, Gold, Ple, dia)
- 1-6 합격 예상 메시지
  80% ~ : 합격 유력
  70% ~ 79% : 합격 예상
  60% ~ 69% : 불합격 예상
  59% : 불합격 유력
- 1-7 직렬 일정은 직렬 당 5개

2. PM과 대화 사항
- 2-1 대시보드 동일 직렬 지원자 수 증가가 무의미하게 작을 것 -> 삭제 요청 or 인원 수로 표현
- 2-2 대시보드 동일 직렬 점수 입력 증가가 무의미하게 작을 것 -> 삭제 요청 or 인원 수로 표현
- 2-3 동일 직렬 점수 입력이 모호하니 제거 요청
- 2-4 동일 직렬 점수는 매달 리셋? or 매년 리셋
- 2-5 지원자 이번 달 점수 -> 최대? 마지막? 평균?
- 2-6 지원자의 키랑 몸무게 필요한가?

3. 페이지를 보면서 필요한 api 목록
- 3-1 대시보드
```
직렬별 지원자 수
직렬별 점수 입력 (일단 직렬별 단순 입력 기입)
지원자 이번 달 점수
지원자 저번 달 점수
직렬별 이번 달 예상 도달 점수
직렬별 저번 달 예상 도달 점수
직렬별 이번 달 랭킹 5위
```
- 3-2 측정 기록
```
지원자별 현재 직렬, 입력 점수들 (달별 평균)
지원자별 현재 직렬, 예상 점수들
지원자 현재 직렬 시험 종목 목록
지원자 현재 직렬 시험 종목 환산 점수
지원자 현재 직렬 백분위
지원자 현재 직렬 백분위에 따른 티어
지원자 현재 직렬 예상 통과 점수 (상위 20%사람의 점수)
```
- 3-3 개인 정보
```
지원자 이름
지원자 직렬
지원자 키
지원자 몸무게
지원자 현재 직렬 현재 점수 (이 달 평균 점수)
지원자 현재 직렬 최고 점수 (전체 최고 점수)
지원자 현재 직렬 종목별 현재 점수 (이 달 종목별 평균 점수)
지원자 현재 직렬 종목별 예상 점수
지원자 현재 직렬 종목별 평균 점수
지원자 현재 직렬 종목별 상위 30퍼 점수
```

- 3-4 설정 - 계정
```
지원자 이름 조회
지원자 이메일 조회
지원자 성별 조회
지원자 비밀번호 변경
지원자 직렬 변경
```

- 3-5 로그인
```
재로그인 없는 로그인 api
재로그인 있는 로그인 api
```

- 3-6 회원가입
```
프론트엔드 수정
회원가입 api
```

- 3-7 비밀번호 찾기
```
메일 임시 메일 전송
```

4. 도메인

유저 (Member) Y
```
ID
name
email
pwd
Gender
ApplicationType
created_at
updated_at
```

종목 (JudgingItem)	Y
```
ID
ApplicationType
name
```

종목 환점	Y
```
ID
judingItem_id
target_score
taken_score
```

점수 (ScoreRecord) J
```
ID
member_id
judgingItem_id
score
created_at
```

직렬 정보(ApplicationType - enum) J
```
standardName
iconImageUrl (“static/default.jpg”)
```
