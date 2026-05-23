const API = window.location.protocol === 'file:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000' : 'https://sanz-ai-tutor.onrender.com';
const GRADE_P_ICONS={1:'🌱',2:'⭐',3:'🌟',4:'🚀',5:'🏆'};
const CREATOR_MSGS={
  sanz:{si:'✨ "Sanz" — Sanduni ගේ හදවතෙන් හදාපු ශ්‍රී ලංකාවේ ස්මාර්ට්ම AI Tutor! 💙🎓',ta:'✨ "Sanz" — Sanduni அவர்களின் இதயத்தால் AI Tutor! 💙🎓',en:'✨ "Sanz" — Built from the heart by Sanduni! 💙🎓'},
  sanduni:{si:'👑💜 Sanduni — Sanz AI ගේ නිර්මාතෘවරිය! 🌸🎓',ta:'👑💜 Sanduni — Sanz AI-இன் படைப்பாளி! 🌸🎓',en:'👑💜 Sanduni — The creator of Sanz AI! 🌸🎓'}
};
const UI={
  si:{greet:(n,g)=>`ආයුබෝවන් **${n}**! 🎓 Grade ${g} ශිෂ්‍යයෙකු ලෙස **Sanz AI** වෙත සාදරයෙන් පිළිගනිමු.\n\nඕනෑ **විෂයයක්** ගැන නිදහසේ අහන්න! 😊`,wch:'ආයුබෝවන්! 👋',wcp:'ඔබේ ප්‍රශ්නය ටයිප් කරන්න',ph:'ප්‍රශ්නය ලියන්න...',listen:'🔊 ඇසීමට',copy:'📋 Copy',copied:'Copied ✓',read:'📖 කියවන්න',empty:'ප්‍රශ්නයක් ලිවීමට අමතකද?',sp:'si-LK',subjects:'විෂයයන්',langLabel:'🇱🇰 සිංහල',catL:(g)=>g<=5?`Grade ${g} · Primary`:g<=11?`Grade ${g} · O/L`:`Grade ${g} · A/L`,catCls:(g)=>g<=5?'bp':g<=11?'bs':'ba',reactLabels:{good:'ලස්සනයි! 👍',ok:'හරි 😊',bad:'නොතේරුණා 😕',ask:'නැවත 🔁'},rpTitle:'📖 කියවීම',sucTitle:'Account හදා ගත්තා! 🎉',sucMsg:'Login details save කරගන්න!',sucBtn:'Login ෙකෙ යන්න',noAcc:'Account නෑද?'},
  ta:{greet:(n,g)=>`வணக்கம் **${n}**! 🎓 Grade ${g} மாணவராக **Sanz AI**-க்கு வரவேற்கிறோம்.\n\nஎந்த **பாடத்தை** பற்றியும் கேளுங்கள்! 😊`,wch:'வணக்கம்! 👋',wcp:'கேள்வியை உள்ளிடுக',ph:'கேள்வியை உள்ளிடுக...',listen:'🔊 கேட்க',copy:'📋 நகலெடு',copied:'நகலெடுத்தது ✓',read:'📖 படிக்கவும்',empty:'கேள்வியை உள்ளிடவும்!',sp:'ta-IN',subjects:'பாடங்கள்',langLabel:'🇱🇰 தமிழ்',catL:(g)=>g<=5?`Grade ${g} · Primary`:g<=11?`Grade ${g} · O/L`:`Grade ${g} · A/L`,catCls:(g)=>g<=5?'bp':g<=11?'bs':'ba',reactLabels:{good:'சிறந்தது! 👍',ok:'சரி 😊',bad:'புரியவில்லை 😕',ask:'மீண்டும் 🔁'},rpTitle:'📖 வாசிப்பு',sucTitle:'கணக்கு உருவாக்கப்பட்டது! 🎉',sucMsg:'Login details சேமிக்கவும்!',sucBtn:'Login செல்',noAcc:'கணக்கு இல்லையா?'},
  en:{greet:(n,g)=>`Hello **${n}**! 🎓 Welcome to **Sanz AI** as a Grade ${g} student.\n\nFeel free to ask about any **subject**! 😊`,wch:'Welcome! 👋',wcp:'Type your question',ph:'Type your question...',listen:'🔊 Listen',copy:'📋 Copy',copied:'Copied ✓',read:'📖 Read',empty:'Please type a question!',sp:'en-US',subjects:'SUBJECTS',langLabel:'🇬🇧 English',catL:(g)=>g<=5?`Grade ${g} · Primary`:g<=11?`Grade ${g} · O/L`:`Grade ${g} · A/L`,catCls:(g)=>g<=5?'bp':g<=11?'bs':'ba',reactLabels:{good:'Helpful! 👍',ok:'Okay 😊',bad:"Didn't understand 😕",ask:'Explain again 🔁'},rpTitle:'📖 Reading',sucTitle:'Account Created! 🎉',sucMsg:'Save your login details!',sucBtn:'Go to Login',noAcc:"Don't have an account?"}
};

let chosenLang='si',currentGrp='p',currentGrade=3;
let S={name:'',grade:6,cat:'secondary',lang:'si',subject:'Mathematics',subIc:'📐',alStream:'bio',rec:false,loading:false};
let conversationMessages=[],msgCounter=0;

// Bubbles
(function(){const c=document.getElementById('lsBubbles');for(let i=0;i<18;i++){const b=document.createElement('div');b.className='ls-bub';const sz=30+Math.random()*120;b.style.cssText=`width:${sz}px;height:${sz}px;left:${Math.random()*100}%;bottom:${-sz}px;animation-duration:${7+Math.random()*11}s;animation-delay:${Math.random()*8}s;`;c.appendChild(b);}})();

// Grade buttons
function buildGradeButtons(){
  const pb=document.getElementById('gbtns-p');if(pb){pb.innerHTML='';for(let g=1;g<=5;g++){const btn=document.createElement('button');btn.className='gb-p'+(g===3?' sel':'');btn.innerHTML=`<span class="gb-num">${g}</span><span class="gb-ico">${GRADE_P_ICONS[g]}</span>`;btn.onclick=()=>selectGrade(g,'p');pb.appendChild(btn);}}
  const sb=document.getElementById('gbtns-s');if(sb){sb.innerHTML='';for(let g=6;g<=11;g++){const btn=document.createElement('button');btn.className='gb'+(g===8?' sel-s':'');btn.textContent=g;btn.onclick=()=>selectGrade(g,'s');sb.appendChild(btn);}}
  const ab=document.getElementById('gbtns-a');if(ab){ab.innerHTML='';for(let g=12;g<=13;g++){const btn=document.createElement('button');btn.className='gb'+(g===12?' sel-a':'');btn.textContent=g;btn.onclick=()=>selectGrade(g,'a');ab.appendChild(btn);}}
}

function selectGrade(g,grp){currentGrade=g;currentGrp=grp;
  if(grp==='p'){document.querySelectorAll('#gbtns-p .gb-p').forEach(b=>{b.classList.toggle('sel',parseInt(b.querySelector('.gb-num').textContent)===g);});}
  else{const sc=grp==='s'?'sel-s':'sel-a';document.querySelectorAll(`#gbtns-${grp} .gb`).forEach(b=>{b.className='gb'+(parseInt(b.textContent)===g?' '+sc:'');});}
}
function switchGrp(grp){currentGrp=grp;['p','s','a'].forEach(x=>{document.getElementById('gtab-'+x).className='grade-tab'+(x===grp?` active-${x}`:'');document.getElementById('gpanel-'+x).className='grade-panel'+(x===grp?' show':'');});const def={p:3,s:8,a:12};selectGrade(def[grp],grp);currentGrade=def[grp];}

function applyTheme(g){document.body.classList.remove('theme-primary','theme-secondary','theme-al');if(g<=5)document.body.classList.add('theme-primary');else if(g<=11)document.body.classList.add('theme-secondary');else document.body.classList.add('theme-al');}

function showScreen(id){document.querySelectorAll('.screen').forEach(s=>s.classList.remove('show'));const el=document.getElementById(id);if(el)el.classList.add('show');}

function chooseLang(lang){
  chosenLang=lang;S.lang=lang;
  document.getElementById('langScreen').style.display='none';
  applyLangUI(lang);
  showScreen('scrChoice');
}

function applyLangUI(lang){
  const L={
    si:{
      wcH:'ආයුබෝවන්! 👋', wcP:'ඔබේ ප්‍රශ්නය ටයිප් කරන්න',
      ph:'ප්‍රශ්නය ලියන්න...', langLabel:'🇱🇰 සිංහල', subjects:'විෂයයන්',
      choiceSub:'ඔබේ account ෙකෙ login කරන්න හෝ නව account හදන්න',
      c1:'Sign Up', c1s:'නව account හදන්න',
      c2:'Sign In', c2s:'ඔයාගෙ account ෙකෙ login කරන්න',
      c3:'Guest', c3s:'Account නැතිව continue කරන්න',
      suTitle:'Sign Up', suSub:'නව account හදන්න',
      suLbNm:'සම්පූර්ණ නම', suNmPh:'e.g. Kasun Silva',
      suLbUs:'Username', suUsPh:'e.g. kasun2010',
      suLbBd:'උපන් දිනය', suLbGr:'ශ්‍රේණිය', suGrPh:'ශ්‍රේණිය තෝරන්න',
      suLbEm:'Email (Progress report ෙකෙ)', suEmPh:'kasun@gmail.com',
      suLbPh:'Phone (optional)', suPhPh:'07X XXXXXXX',
      suLbPw:'Password', suPwPh:'අවම වශයෙන් 6 characters',
      suLbP2:'Confirm Password', suP2Ph:'Password නැවත ලියන්න',
      suBtn:'Account හදන්න',
      sucTitle:'Account හදා ගත්තා! 🎉',
      sucMsg:'ඔයාගේ login details save කරගන්න! Screenshot ගන්න!',
      sucBtn:'Login ෙකෙ යන්න',
      siTitle:'Sign In', siSub:'ඔයාගේ account ෙකෙ login කරන්න',
      siLbUs:'Username', siUsPh:'ඔයාගෙ username',
      siLbPw:'Password', siPwPh:'••••••••',
      siBtn:'Login', noAcc:'Account නෑද? Sign Up කරන්න',
    },
    ta:{
      wcH:'வணக்கம்! 👋', wcP:'கேள்வியை உள்ளிடுக',
      ph:'கேள்வியை உள்ளிடுக...', langLabel:'🇱🇰 தமிழ்', subjects:'பாடங்கள்',
      choiceSub:'உங்கள் கணக்கில் உள்நுழையுங்கள் அல்லது புதியது உருவாக்குங்கள்',
      c1:'Sign Up', c1s:'புதிய கணக்கு உருவாக்குங்கள்',
      c2:'Sign In', c2s:'உங்கள் கணக்கில் உள்நுழையுங்கள்',
      c3:'Guest', c3s:'கணக்கு இல்லாமல் தொடரவும்',
      suTitle:'Sign Up', suSub:'புதிய கணக்கு உருவாக்குங்கள்',
      suLbNm:'முழு பெயர்', suNmPh:'e.g. Kaviya Silva',
      suLbUs:'பயனர் பெயர்', suUsPh:'e.g. kaviya2010',
      suLbBd:'பிறந்த தேதி', suLbGr:'வகுப்பு', suGrPh:'வகுப்பை தேர்ந்தெடுக்கவும்',
      suLbEm:'Email (அறிக்கைக்கு)', suEmPh:'kaviya@gmail.com',
      suLbPh:'தொலைபேசி (விருப்பமானது)', suPhPh:'07X XXXXXXX',
      suLbPw:'கடவுச்சொல்', suPwPh:'குறைந்தது 6 எழுத்துக்கள்',
      suLbP2:'கடவுச்சொல் உறுதிப்படுத்தவும்', suP2Ph:'மீண்டும் உள்ளிடவும்',
      suBtn:'கணக்கு உருவாக்கு',
      sucTitle:'கணக்கு உருவாக்கப்பட்டது! 🎉',
      sucMsg:'உங்கள் login விவரங்களை சேமிக்கவும்! Screenshot எடுங்கள்!',
      sucBtn:'Login க்கு செல்',
      siTitle:'Sign In', siSub:'உங்கள் கணக்கில் உள்நுழையுங்கள்',
      siLbUs:'பயனர் பெயர்', siUsPh:'உங்கள் username',
      siLbPw:'கடவுச்சொல்', siPwPh:'••••••••',
      siBtn:'உள்நுழை', noAcc:'கணக்கு இல்லையா? Sign Up செய்யுங்கள்',
    },
    en:{
      wcH:'Welcome! 👋', wcP:'Type your question',
      ph:'Type your question...', langLabel:'🇬🇧 English', subjects:'SUBJECTS',
      choiceSub:'Login to your account or create a new one',
      c1:'Sign Up', c1s:'Create a new account',
      c2:'Sign In', c2s:'Login to your account',
      c3:'Guest', c3s:'Continue without an account',
      suTitle:'Sign Up', suSub:'Create a new account',
      suLbNm:'Full Name', suNmPh:'e.g. Kasun Silva',
      suLbUs:'Username', suUsPh:'e.g. kasun2010',
      suLbBd:'Birthday', suLbGr:'Grade', suGrPh:'Select your grade',
      suLbEm:'Email (for progress reports)', suEmPh:'kasun@gmail.com',
      suLbPh:'Phone Number (optional)', suPhPh:'07X XXXXXXX',
      suLbPw:'Password', suPwPh:'Minimum 6 characters',
      suLbP2:'Confirm Password', suP2Ph:'Re-enter your password',
      suBtn:'Create Account',
      sucTitle:'Account Created! 🎉',
      sucMsg:'Save your login details! Take a screenshot!',
      sucBtn:'Go to Login',
      siTitle:'Sign In', siSub:'Login to your account',
      siLbUs:'Username', siUsPh:'Your username',
      siLbPw:'Password', siPwPh:'••••••••',
      siBtn:'Login', noAcc:"Don't have an account? Sign Up",
    }
  };
  const l=L[lang]||L.si;

  // Welcome card & chat
  document.getElementById('wcH').textContent=l.wcH;
  document.getElementById('wcP').textContent=l.wcP;
  document.getElementById('qIn').placeholder=l.ph;
  document.getElementById('tbLb').textContent=l.langLabel;
  document.getElementById('sbSecLbl').textContent=l.subjects;

  // Choice screen
  const choiceSub=document.getElementById('choiceSub');if(choiceSub)choiceSub.textContent=l.choiceSub;
  const c1t=document.getElementById('c1t');if(c1t)c1t.textContent=l.c1;
  const c1s=document.getElementById('c1s');if(c1s)c1s.textContent=l.c1s;
  const c2t=document.getElementById('c2t');if(c2t)c2t.textContent=l.c2;
  const c2s=document.getElementById('c2s');if(c2s)c2s.textContent=l.c2s;
  const c3t=document.getElementById('c3t');if(c3t)c3t.textContent=l.c3;
  const c3s=document.getElementById('c3s');if(c3s)c3s.textContent=l.c3s;

  // Sign Up screen — labels & placeholders
  const suTi=document.getElementById('suTitle');if(suTi)suTi.textContent=l.suTitle;
  const suSb=document.getElementById('suSub');if(suSb)suSb.textContent=l.suSub;
  const suLbNm=document.getElementById('suLbNm');if(suLbNm)suLbNm.textContent=l.suLbNm;
  const suName=document.getElementById('suName');if(suName)suName.placeholder=l.suNmPh;
  const suLbUs=document.getElementById('suLbUs');if(suLbUs)suLbUs.textContent=l.suLbUs;
  const suUser=document.getElementById('suUser');if(suUser)suUser.placeholder=l.suUsPh;
  const suLbBd=document.getElementById('suLbBd');if(suLbBd)suLbBd.textContent=l.suLbBd;
  const suLbGr=document.getElementById('suLbGr');if(suLbGr)suLbGr.textContent=l.suLbGr;
  const suGrade=document.getElementById('suGrade');if(suGrade)suGrade.options[0].textContent=l.suGrPh;
  const suLbEm=document.getElementById('suLbEm');if(suLbEm)suLbEm.textContent=l.suLbEm;
  const suEmail=document.getElementById('suEmail');if(suEmail)suEmail.placeholder=l.suEmPh;
  const suLbPh=document.getElementById('suLbPh');if(suLbPh)suLbPh.textContent=l.suLbPh;
  const suPhone=document.getElementById('suPhone');if(suPhone)suPhone.placeholder=l.suPhPh;
  const suLbPw=document.getElementById('suLbPw');if(suLbPw)suLbPw.textContent=l.suLbPw;
  const suPw=document.getElementById('suPw');if(suPw)suPw.placeholder=l.suPwPh;
  const suLbP2=document.getElementById('suLbP2');if(suLbP2)suLbP2.textContent=l.suLbP2;
  const suPw2=document.getElementById('suPw2');if(suPw2)suPw2.placeholder=l.suP2Ph;
  const suBtnTx=document.getElementById('suBtnTx');if(suBtnTx)suBtnTx.textContent=l.suBtn;

  // Success screen
  const sucTi=document.getElementById('sucTitle');if(sucTi)sucTi.textContent=l.sucTitle;
  const sucMs=document.getElementById('sucMsg');if(sucMs)sucMs.textContent=l.sucMsg;
  const sucBt=document.getElementById('sucBtnTx');if(sucBt)sucBt.textContent=l.sucBtn;

  // Login screen
  const siTi=document.getElementById('siTitle');if(siTi)siTi.textContent=l.siTitle;
  const siSb=document.getElementById('siSub');if(siSb)siSb.textContent=l.siSub;
  const siLbUs=document.getElementById('siLbUs');if(siLbUs)siLbUs.textContent=l.siLbUs;
  const siUser=document.getElementById('siUser');if(siUser)siUser.placeholder=l.siUsPh;
  const siLbPw=document.getElementById('siLbPw');if(siLbPw)siLbPw.textContent=l.siLbPw;
  const siPw=document.getElementById('siPw');if(siPw)siPw.placeholder=l.siPwPh;
  const siBtnTx=document.getElementById('siBtnTx');if(siBtnTx)siBtnTx.textContent=l.siBtn;
  const noAccLink=document.getElementById('noAccLink');if(noAccLink)noAccLink.textContent=l.noAcc;

  // Warn note on success screen
  const warnNote=document.getElementById('warnNote');
  if(warnNote){
    const wt={si:'⚠️ <strong>වැදගත්!</strong> Username සහ password ලියාගන්න හෝ screenshot ගන්න. Password recover කරන්න බෑ!',ta:'⚠️ <strong>முக்கியம்!</strong> Username மற்றும் password-ஐ எழுதி வையுங்கள் அல்லது screenshot எடுங்கள். Password மீட்டெடுக்க முடியாது!',en:'⚠️ <strong>Important!</strong> Write down your username and password or take a screenshot. Password cannot be recovered!'};
    warnNote.innerHTML=wt[lang]||wt.si;
  }
}

function togglePw(id,btn){const i=document.getElementById(id);i.type=i.type==='password'?'text':'password';btn.textContent=i.type==='password'?'👁':'🙈';}

function onNameType(){
  const nm=(document.getElementById('suUser').value||'').toLowerCase();
  const banner=document.getElementById('creatorBanner');
  let msg='',type='';
  if(nm.includes('sanduni')){msg=CREATOR_MSGS.sanduni[chosenLang];type='type-sanduni';}
  else if(nm.includes('sanz')){msg=CREATOR_MSGS.sanz[chosenLang];type='type-sanz';}
  if(msg){banner.textContent=msg;banner.className='creator-banner show '+type;}
  else{banner.className='creator-banner';}
}

// ══ SIGN UP ══
async function doSignup(){
  const name=document.getElementById('suName').value.trim();
  const user=document.getElementById('suUser').value.trim().toLowerCase().replace(/\s+/g,'_');
  const bd=document.getElementById('suBd').value;
  const grade=document.getElementById('suGrade').value;
  const email=document.getElementById('suEmail').value.trim();
  const phone=document.getElementById('suPhone').value.trim();
  const pw=document.getElementById('suPw').value;
  const pw2=document.getElementById('suPw2').value;
  const errEl=document.getElementById('suErr');errEl.className='form-err';
  const EM={
    si:{name:'⚠️ සම්පූර්ණ නම ඇතුළත් කරන්න!',user:'⚠️ Username අවම 3 characters ඕනෙ!',bd:'⚠️ උපන් දිනය ඇතුළත් කරන්න!',grade:'⚠️ ශ්‍රේණිය තෝරන්න!',contact:'⚠️ Email හෝ Phone number ඕනෙ!',pw:'⚠️ Password අවම 6 characters ඕනෙ!',pw2:'⚠️ Passwords match නෑ!',fail:'❌ Registration failed',server:'⚠️ Server connect නොවිය!'},
    ta:{name:'⚠️ முழு பெயர் தேவை!',user:'⚠️ Username குறைந்தது 3 எழுத்துக்கள் தேவை!',bd:'⚠️ பிறந்த தேதி தேவை!',grade:'⚠️ வகுப்பை தேர்ந்தெடுக்கவும்!',contact:'⚠️ Email அல்லது Phone தேவை!',pw:'⚠️ Password குறைந்தது 6 எழுத்துக்கள்!',pw2:'⚠️ Passwords பொருந்தவில்லை!',fail:'❌ பதிவு தோல்வியடைந்தது',server:'⚠️ Server இணைக்க முடியவில்லை!'},
    en:{name:'⚠️ Full name required!',user:'⚠️ Username must be 3+ characters!',bd:'⚠️ Birthday required!',grade:'⚠️ Please select your grade!',contact:'⚠️ Email or phone number required!',pw:'⚠️ Password must be 6+ characters!',pw2:'⚠️ Passwords do not match!',fail:'❌ Registration failed',server:'⚠️ Server connect failed!'}
  };
  const em=EM[chosenLang]||EM.si;
  if(!name){errEl.textContent=em.name;errEl.className='form-err show';return;}
  if(!user||user.length<3){errEl.textContent=em.user;errEl.className='form-err show';return;}
  if(!bd){errEl.textContent=em.bd;errEl.className='form-err show';return;}
  if(!grade){errEl.textContent=em.grade;errEl.className='form-err show';return;}
  if(!email&&!phone){errEl.textContent=em.contact;errEl.className='form-err show';return;}
  if(!pw||pw.length<6){errEl.textContent=em.pw;errEl.className='form-err show';return;}
  if(pw!==pw2){errEl.textContent=em.pw2;errEl.className='form-err show';return;}
  const btn=document.getElementById('suBtn');btn.disabled=true;
  try{
    const res=await fetch(`${API}/user/register`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:user,password:pw,full_name:name,birthday:bd,grade,email,phone,language:chosenLang})});
    const d=await res.json();
    if(!res.ok){errEl.textContent='❌ '+(d.detail||'Registration failed');errEl.className='form-err show';btn.disabled=false;return;}
    const t=UI[chosenLang];
    document.getElementById('sucTitle').textContent=t.sucTitle;
    document.getElementById('sucMsg').textContent=t.sucMsg;
    document.getElementById('sucUser').textContent=d.username;
    document.getElementById('sucPw').textContent=d.password;
    document.getElementById('sucGrade').textContent='Grade '+grade;
    document.getElementById('sucEmail').textContent=email||phone||'—';
    showScreen('scrSuccess');
  }catch(e){errEl.textContent='⚠️ Server connect failed!';errEl.className='form-err show';}
  btn.disabled=false;
}

function goToLogin(){document.getElementById('siUser').value=document.getElementById('sucUser').textContent;showScreen('scrLogin');}

// ══ LOGIN ══
async function doLogin(){
  const user=document.getElementById('siUser').value.trim().toLowerCase().replace(/\s+/g,'_');
  const pw=document.getElementById('siPw').value;
  const errEl=document.getElementById('siErr');const btn=document.getElementById('siBtn');
  errEl.className='form-err';
  const LEM={si:'⚠️ සියලු fields පුරවන්න!',ta:'⚠️ அனைத்து புலங்களையும் நிரப்பவும்!',en:'⚠️ Fill all fields!'};
  if(!user||!pw){errEl.textContent=LEM[chosenLang]||LEM.si;errEl.className='form-err show';return;}
  btn.disabled=true;
  try{
    const res=await fetch(`${API}/user/login`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:user,password:pw})});
    const d=await res.json();
    if(!res.ok){errEl.textContent='❌ '+(d.detail||'Login failed');errEl.className='form-err show';btn.disabled=false;return;}
    startSession(d.username,d.full_name,parseInt(d.grade)||6,d.language||chosenLang);
  }catch(e){errEl.textContent='⚠️ Server connect failed!';errEl.className='form-err show';}
  btn.disabled=false;
}

function guestMode(){
  const gName='Guest_'+Math.floor(Math.random()*9000+1000);
  startSession(gName,gName,8,chosenLang);
}

function startSession(username,fullName,grade,lang){
  S.name=username;S.lang=lang;S.grade=grade;
  S.cat=grade<=5?'primary':grade<=11?'secondary':'al';
  applyTheme(grade);applyLangUI(lang);
  const t=UI[lang];
  const catL=t.catL(grade);const catCls=t.catCls(grade);
  const sbCls={primary:'p',secondary:'s',al:'a'}[S.cat];
  document.getElementById('tbGb').textContent=catL;document.getElementById('tbGb').className='tb-b '+catCls;
  document.getElementById('sbMeta').innerHTML=`<b>${fullName}</b> &nbsp;•&nbsp; Grade ${grade}`;
  const sc=document.getElementById('sbCat');sc.className='sb-cat '+sbCls;sc.textContent=catL;
  conversationMessages=[];buildSidebar();
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('show'));
  setTimeout(()=>addAI(t.greet(fullName,grade),null,true),320);
  setTimeout(()=>loadProgress(),500);
}

// ══ SUBJECTS ══
const SBJ={
  primary:[{id:'Mathematics',ic:'📐',en:'Mathematics',si:'ගණිතය',ta:'கணிதம்'},{id:'Environment',ic:'🌿',en:'Environment',si:'පරිසරය',ta:'சூழல்'},{id:'Sinhala',ic:'📖',en:'Sinhala',si:'සිංහල',ta:'සිங்களம்'},{id:'Tamil',ic:'📜',en:'Tamil',si:'දෙමළ',ta:'தமிழ்'},{id:'English',ic:'🌐',en:'English',si:'ඉංග්‍රීසි',ta:'ஆங்கிலம்'},{id:'Art',ic:'🎨',en:'Art',si:'චිත්‍ර',ta:'கலை'},{id:'Religion',ic:'🙏',en:'Religion',si:'ආගම',ta:'சமயம்'},{id:'Health',ic:'💊',en:'Health Education',si:'සෞඛ්‍ය',ta:'உடல் நலம்'}],
  secondary:[{id:'Mathematics',ic:'📐',en:'Mathematics',si:'ගණිතය',ta:'கணிதம்'},{id:'Science',ic:'🔬',en:'Science',si:'විද්‍යාව',ta:'அறிவியல்'},{id:'History',ic:'🏛️',en:'History',si:'ඉතිහාසය',ta:'வரலாறு'},{id:'Geography',ic:'🌍',en:'Geography',si:'භූගෝල',ta:'புவியியல்'},{id:'ICT',ic:'💻',en:'ICT',si:'තොරතුරු',ta:'தகவல்'},{id:'Civic',ic:'⚖️',en:'Civic Education',si:'පුරවැසි',ta:'குடியியல்'},{id:'Sinhala',ic:'📖',en:'Sinhala',si:'සිංහල',ta:'සිங்களம்'},{id:'Tamil',ic:'📜',en:'Tamil',si:'දෙමළ',ta:'தமிழ்'},{id:'English',ic:'🌐',en:'English',si:'ඉංග්‍රීසි',ta:'ஆங்கிலம்'},{id:'Commerce',ic:'📊',en:'Commerce',si:'වාණිජ',ta:'வணிகவியல்'},{id:'Health',ic:'💊',en:'Health & P.E.',si:'සෞඛ්‍ය',ta:'உடல் கல்வி'},{id:'Music',ic:'🎵',en:'Music',si:'සංගීතය',ta:'இசை'},{id:'Drama',ic:'🎭',en:'Drama',si:'නාට්‍ය',ta:'நாடகம்'}],
  alStreams:[{id:'bio',ic:'🧬',en:'Biology Stream',si:'ජීව ධාරාව',ta:'உயிரியல்',sbj:[{id:'Biology',ic:'🧬',en:'Biology',si:'ජීව',ta:'உயிரியல்'},{id:'Chemistry',ic:'⚗️',en:'Chemistry',si:'රසායන',ta:'வேதியியல்'},{id:'Physics',ic:'⚡',en:'Physics',si:'භෞතික',ta:'இயற்பியல்'}]},{id:'maths',ic:'➕',en:'Maths Stream',si:'ගණිත ධාරාව',ta:'கணிதம்',sbj:[{id:'CombinedMaths',ic:'📐',en:'Combined Maths',si:'සංයුක්ත',ta:'இணைந்த'},{id:'Physics',ic:'⚡',en:'Physics',si:'භෞතික',ta:'இயற்பியல்'},{id:'Chemistry',ic:'⚗️',en:'Chemistry',si:'රසායන',ta:'வேதியியல்'},{id:'ICT',ic:'💻',en:'ICT',si:'IT',ta:'IT'}]},{id:'commerce',ic:'💼',en:'Commerce Stream',si:'වාණිජ ධාරාව',ta:'வணிகம்',sbj:[{id:'Economics',ic:'📈',en:'Economics',si:'ආර්ථික',ta:'பொருளாதாரம்'},{id:'BusinessStudies',ic:'🏢',en:'Business Studies',si:'ව්‍යාපාර',ta:'வணிக கல்வி'},{id:'Accounting',ic:'🧾',en:'Accounting',si:'ගිණුම්',ta:'கணக்கியல்'},{id:'Statistics',ic:'📊',en:'Statistics',si:'සංඛ්‍යාන',ta:'புள்ளියியல்'}]},{id:'tech',ic:'🔧',en:'Technology Stream',si:'තාක්ෂණ ධාරාව',ta:'தொழில்நுட்பம்',sbj:[{id:'EngineeringTech',ic:'⚙️',en:'Engineering Tech.',si:'ඉංජිනේරු',ta:'பொறியியல்'},{id:'ScienceTech',ic:'🔬',en:'Science for Tech.',si:'තාක්ෂණ',ta:'அறிவியல்'},{id:'ICT',ic:'💻',en:'ICT',si:'IT',ta:'IT'},{id:'Agriculture',ic:'🌾',en:'Agriculture',si:'කෘෂිකර්ම',ta:'விவசாயம்'}]},{id:'art',ic:'🎨',en:'Arts Stream',si:'කලා ධාරාව',ta:'கலை',sbj:[{id:'SinhalaLit',ic:'📖',en:'Sinhala Literature',si:'සිංහල සාහිත්‍ය',ta:'சிங்கள'},{id:'Geography',ic:'🌍',en:'Geography',si:'භූගෝල',ta:'புவியியல்'},{id:'Logic',ic:'🧩',en:'Logic',si:'තර්ක',ta:'தர்க்கம்'},{id:'Art',ic:'🖼️',en:'Art',si:'චිත්‍ර',ta:'கலை'},{id:'Music',ic:'🎵',en:'Music',si:'සංගීතය',ta:'இசை'}]},{id:'lang',ic:'🌐',en:'Languages Stream',si:'භාෂා ධාරාව',ta:'மொழி',sbj:[{id:'Sinhala',ic:'📖',en:'Sinhala',si:'සිංහල',ta:'சிங்களம்'},{id:'Tamil',ic:'📜',en:'Tamil',si:'දෙමළ',ta:'தமிழ்'},{id:'English',ic:'🌐',en:'English',si:'ඉංග්‍රීසි',ta:'ஆங்கிலம்'},{id:'Pali',ic:'📿',en:'Pali/Sanskrit',si:'පාලි',ta:'பாலி'},{id:'French',ic:'🇫🇷',en:'French',si:'ප්‍රංශ',ta:'பிரஞ்சு'}]}],
  alCommon:[{id:'GenEnglish',ic:'🌐',en:'General English',si:'සාමාන්‍ය ඉංග්‍රීසි',ta:'பொது ஆங்கிலம்'},{id:'GenIT',ic:'💻',en:'General IT',si:'සාමාන්‍ය IT',ta:'பொது IT'}]
};

function buildSidebar(){const stSel=document.getElementById('stSel');if(S.cat==='al'){stSel.style.display='block';buildStreams();buildSubjs(getALSubjs(S.alStream));}else{stSel.style.display='none';buildSubjs(SBJ[S.cat]);}}
function buildStreams(){const c=document.getElementById('stBtns');c.innerHTML='';SBJ.alStreams.forEach(st=>{const b=document.createElement('button');b.className='st-btn'+(st.id===S.alStream?' active':'');const nm=S.lang==='ta'?st.ta:S.lang==='si'?st.si:st.en;b.innerHTML=`<div class="st-ico">${st.ic}</div><div><div class="st-nm">${nm||st.en}</div><div class="st-lc">${st.en}</div></div>`;b.onclick=()=>{S.alStream=st.id;buildStreams();buildSubjs(getALSubjs(st.id));};c.appendChild(b);});}
function getALSubjs(sid){const st=SBJ.alStreams.find(s=>s.id===sid);return[...(st?st.sbj:[]),...SBJ.alCommon];}
function buildSubjs(list){const nav=document.getElementById('sbNav');nav.innerHTML='';list.forEach((s,i)=>{const b=document.createElement('button');b.className='sbj'+(i===0?' active':'');const lc=S.lang==='ta'?s.ta:S.lang==='si'?s.si:s.en;b.innerHTML=`<div class="sbj-ic">${s.ic}</div><div class="sbj-tx"><div class="sbj-en">${s.en}</div><div class="sbj-lc">${lc}</div></div>`;b.onclick=()=>{S.subject=s.id;S.subIc=s.ic;conversationMessages=[];document.querySelectorAll('.sbj').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById('tbIc').textContent=s.ic;document.getElementById('tbNm').textContent=`${s.en} / ${s.si||s.en}`;document.getElementById('tbLc').textContent=s.ta||s.si||s.en;};nav.appendChild(b);if(i===0){S.subject=s.id;S.subIc=s.ic;document.getElementById('tbIc').textContent=s.ic;document.getElementById('tbNm').textContent=`${s.en} / ${s.si||s.en}`;document.getElementById('tbLc').textContent=s.ta||s.si||s.en;}});}

// ══ REACTIONS ══
const REACTIONS=[{emoji:'👍',id:'good',cls:'reacted'},{emoji:'😊',id:'ok',cls:'reacted'},{emoji:'😕',id:'bad',cls:'reacted-bad'},{emoji:'🔁',id:'ask',cls:'reacted'}];
function buildReactions(msgId){const lang=S.lang;const labels=UI[lang].reactLabels;const wrap=document.createElement('div');wrap.className='reactions-wrap';const lbl=document.createElement('div');lbl.className='react-label-row';lbl.textContent={si:'ඔබේ ප්‍රතිචාරය:',ta:'உங்கள் கருத்து:',en:'Your reaction:'}[lang];wrap.appendChild(lbl);const row=document.createElement('div');row.className='react-row';row.id=`react-${msgId}`;REACTIONS.forEach(r=>{const btn=document.createElement('button');btn.className='react-btn';btn.dataset.id=r.id;btn.dataset.cls=r.cls;const cnt=document.createElement('span');cnt.className='react-count';cnt.id=`rcnt-${msgId}-${r.id}`;const txt=document.createElement('span');txt.className='react-txt';txt.textContent=labels[r.id]||r.id;btn.appendChild(document.createTextNode(r.emoji+' '));btn.appendChild(txt);btn.appendChild(cnt);btn.onclick=()=>doReact(msgId,r,btn);row.appendChild(btn);});wrap.appendChild(row);return wrap;}
function doReact(msgId,reaction,btn){const row=document.getElementById(`react-${msgId}`);const already=row.querySelector('.reacted,.reacted-bad');if(already===btn){btn.className='react-btn';const c=document.getElementById(`rcnt-${msgId}-${reaction.id}`);const v=parseInt(c.textContent)||0;c.textContent=v>1?v-1:'';return;}if(already){const oid=already.dataset.id;already.className='react-btn';const oc=document.getElementById(`rcnt-${msgId}-${oid}`);if(oc){const v=parseInt(oc.textContent)||0;oc.textContent=v>1?v-1:'';}}btn.className='react-btn '+reaction.cls+' react-pop';setTimeout(()=>btn.classList.remove('react-pop'),400);const c=document.getElementById(`rcnt-${msgId}-${reaction.id}`);c.textContent=(parseInt(c.textContent)||0)+1;const lang=S.lang;if(reaction.id==='ask'){const ra={si:'කරුණාකර මෙය සරලව නැවත පැහැදිලි කරන්න.',ta:'இதை மீண்டும் எளிமையாக விளக்குங்கள்.',en:'Please explain this again more simply.'};document.getElementById('qIn').value=ra[lang];aH(document.getElementById('qIn'));}const goodMsg={si:'🌟 ලස්සනයි! ඉගෙනීම ඉදිරියට!',ta:'🌟 சிறந்தது!',en:'🌟 Great! Keep learning!'};if(reaction.id==='good')showToast(goodMsg[lang]);}

// ══ IMAGE ══
function onImgSel(e){const f=e.target.files[0];if(!f)return;document.getElementById('ipImg').src=URL.createObjectURL(f);document.getElementById('ipWrap').style.display='inline-block';document.getElementById('uplLbl').classList.add('hf');}
function rmImg(){document.getElementById('imgInp').value='';document.getElementById('ipWrap').style.display='none';document.getElementById('uplLbl').classList.remove('hf');}

// ══ VOICE ══
let recog=null;
if('webkitSpeechRecognition' in window||'SpeechRecognition' in window){const SR=window.SpeechRecognition||window.webkitSpeechRecognition;recog=new SR();recog.continuous=true;recog.interimResults=true;recog.onresult=e=>{let t='';for(let i=0;i<e.results.length;i++)t+=e.results[i][0].transcript;const ta=document.getElementById('qIn');ta.value=t;aH(ta);};recog.onend=()=>{if(S.rec)recog.start();};}
function toggleMic(){if(!recog){showToast('Voice not supported on this browser');return;}const b=document.getElementById('micBtn');if(S.rec){recog.stop();S.rec=false;b.textContent='🎤';b.classList.remove('rec');}else{recog.lang=UI[S.lang].sp;recog.start();S.rec=true;b.textContent='⏹';b.classList.add('rec');const lm={si:'🎤 සවන් දෙමින්...',ta:'🎤 கேட்கிறேன்...',en:'🎤 Listening...'};showToast(lm[S.lang]);}}
function speak(txt){window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(txt.replace(/[*_#`]/g,''));u.lang=UI[S.lang].sp;u.rate=.93;window.speechSynthesis.speak(u);}

// ══ SEND ══
async function sendMsg(){
  const inp=document.getElementById('qIn');const q=inp.value.trim();
  const t=UI[S.lang];if(!q){showToast(t.empty);return;}if(S.loading)return;
  const wc=document.getElementById('wcCard');if(wc)wc.remove();
  addUser(q);inp.value='';aH(inp);
  conversationMessages.push({role:'user',text:q});
  const imgFile=document.getElementById('imgInp').files[0]||null;rmImg();showTyp();setLoad(true);
  try{
    const fd=new FormData();
    fd.append('name',S.name);fd.append('grade',S.grade);fd.append('subject',S.subject);
    fd.append('question',q);fd.append('language',S.lang);fd.append('is_crosscheck','false');
    if(imgFile)fd.append('image',imgFile);
    fd.append('conversation_history',JSON.stringify(conversationMessages));
    const res=await fetch(`${API}/solve`,{method:'POST',body:fd});
    const d=await res.json();hideTyp();setLoad(false);
    if(d.status==='banned'){addAI(d.answer||'⛔ Blocked.',null,false);}
    else if(d.status==='rate_limit'){
      addAI(d.answer,null,false);
      const rm={si:'🔄 තත්පර 10කින් auto-retry...',ta:'🔄 10 விநாடிகளில் மீண்டும்...',en:'🔄 Auto-retrying in 10s...'};
      showToast(rm[S.lang],9500);
      setTimeout(()=>{document.getElementById('qIn').value=q;aH(document.getElementById('qIn'));sendMsg();},10000);
    }
    else if(d.status==='success'){
      addAI(d.answer,d.graph_url,true);
      conversationMessages.push({role:'ai',text:d.answer});
      if(d.progress)updateXpDisplay(d.progress);
      if(d.new_badge)showToast('🎉 New Badge: '+d.new_badge,3500);
    }
    else addAI('⚠️ '+(d.message||'Error'),null,false);
  }catch(e){hideTyp();setLoad(false);addAI({si:'⚠️ Server සමඟ සම්බන්ධ වීමට නොහැකි.',ta:'⚠️ சர்வரில் பிழை.',en:'⚠️ Could not connect.'}[S.lang],null,false);}
}

// ══ MESSAGES ══
function addUser(txt){const c=document.getElementById('chatArea'),w=document.createElement('div');w.className='mw user';w.innerHTML=`<div class="mav">👤</div><div class="mbody"><div class="mbub">${esc(txt)}</div></div>`;c.appendChild(w);se();}

function addAI(md,graph,withReactions){
  const c=document.getElementById('chatArea'),w=document.createElement('div');w.className='mw ai';
  const t=UI[S.lang];const safe=(md||'').replace(/\\/g,'\\\\').replace(/`/g,"'");const mid=++msgCounter;
  const gr=graph?`<img src="${graph}" class="m-graph" alt="Graph">`:'';
  w.innerHTML=`<div class="mav">🎓</div>
    <div class="mbody">
      <div class="mbub">${parseMD(md||'')}${gr}</div>
      <div class="macts">
        <button class="mact" onclick="speak(this.closest('.mbody').querySelector('.mbub').innerText)">${t.listen}</button>
        <button class="mact" onclick="cpyMsg(this,\`${safe}\`)">${t.copy}</button>
        <button class="mact read" onclick="openReadPage(this)">${t.read}</button>
      </div>
    </div>`;
  if(withReactions)w.querySelector('.mbody').appendChild(buildReactions(mid));
  c.appendChild(w);
  try{const bub=w.querySelector('.mbub');if(window.renderMathInElement)renderMathInElement(bub,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false},{left:'\\(',right:'\\)',display:false},{left:'\\[',right:'\\]',display:true}],throwOnError:false});}catch(e){}
  se();
}

let typEl=null;
function showTyp(){hideTyp();const c=document.getElementById('chatArea');typEl=document.createElement('div');typEl.id='typEl';typEl.className='ty-w';typEl.innerHTML=`<div class="mav" style="background:linear-gradient(135deg,var(--primary),var(--primary-lt));color:#fff;">🎓</div><div class="ty-b"><div class="td"></div><div class="td"></div><div class="td"></div></div>`;c.appendChild(typEl);se();}
function hideTyp(){const e=document.getElementById('typEl');if(e)e.remove();typEl=null;}

// ══ UTILS ══
function setLoad(v){S.loading=v;document.getElementById('sendBtn').disabled=v;}
function se(){const c=document.getElementById('chatArea');requestAnimationFrame(()=>c.scrollTop=c.scrollHeight);}
function aH(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,130)+'px';}
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}}
function esc(s){return s==null?'':String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function cpyMsg(btn,txt){navigator.clipboard.writeText(txt).then(()=>{const o=btn.innerHTML;btn.innerHTML=UI[S.lang].copied;setTimeout(()=>btn.innerHTML=o,2000);});}
function showToast(m,d=2800){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),d);}
function updateXpDisplay(p){if(!p)return;const el=document.getElementById('tbXp');el.style.display='flex';document.getElementById('tbXpVal').textContent=p.xp||0;document.getElementById('tbStreak').textContent=p.streak||0;}

// ══ MARKDOWN ══
function parseMD(md){if(!md)return'';let h=md;h=h.replace(/```[\w]*\n?([\s\S]*?)```/g,'<pre><code>$1</code></pre>');h=h.replace(/`([^`\n]+)`/g,'<code>$1</code>');h=h.replace(/^### (.+)$/gm,'<h3>$1</h3>');h=h.replace(/^## (.+)$/gm,'<h2>$1</h2>');h=h.replace(/^# (.+)$/gm,'<h1>$1</h1>');h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');h=h.replace(/\*(.+?)\*/g,'<em>$1</em>');h=h.replace(/^[-*] (.+)$/gm,'<li>$1</li>');h=h.replace(/^\d+\. (.+)$/gm,'<li>$1</li>');h=h.replace(/((?:<li>.*<\/li>\n?)+)/g,'<ul>$1</ul>');h=h.replace(/^> (.+)$/gm,'<blockquote>$1</blockquote>');h=h.replace(/^---+$/gm,'<hr>');h=h.replace(/\n\n+/g,'</p><p>');h=h.replace(/\n/g,'<br>');h='<p>'+h+'</p>';h=h.replace(/<p>\s*<\/p>/g,'');['pre','h1','h2','h3','ul','hr','blockquote'].forEach(tag=>{h=h.replace(new RegExp(`<p>(<${tag}>)`,'g'),'$1');h=h.replace(new RegExp(`(</${tag}>)<\\/p>`,'g'),'$1');});return h;}

// ══ READ PAGE ══
let rpFontLevel=1;
let rpThemeLevel=0;
let rpCurrentText='';
const rpFonts=['fs-sm','fs-md','fs-lg','fs-xl'];
const rpThemes=['th-white','th-cream','th-dark'];
const rpThemeIcons=['🌙','☀️','🌓'];

function openReadPage(btn){
  const mbub=btn.closest('.mbody').querySelector('.mbub');
  rpCurrentText=mbub.innerText;
  document.getElementById('rpContent').innerHTML=mbub.innerHTML;
  document.getElementById('rpSubject').textContent=S.subject||'Answer';
  try{if(window.renderMathInElement)renderMathInElement(document.getElementById('rpContent'),{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});}catch(e){}
  document.getElementById('rpBar').style.width='0%';
  document.getElementById('readPage').classList.add('open');
}
function closeReadPage(){document.getElementById('readPage').classList.remove('open');window.speechSynthesis.cancel();}
function rpFont(dir){
  rpFontLevel=Math.max(0,Math.min(3,rpFontLevel+dir));
  const rp=document.getElementById('readPage');
  rpFonts.forEach(c=>rp.classList.remove(c));
  rp.classList.add(rpFonts[rpFontLevel]);
}
function rpTheme(){
  rpThemeLevel=(rpThemeLevel+1)%3;
  const rp=document.getElementById('readPage');
  rpThemes.forEach(c=>rp.classList.remove(c));
  rp.classList.add(rpThemes[rpThemeLevel]);
  document.getElementById('rpThemeBtn').textContent=rpThemeIcons[rpThemeLevel];
}
function rpSpeak(){window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(rpCurrentText.replace(/[*_#`]/g,''));u.lang=UI[S.lang].sp;u.rate=.93;window.speechSynthesis.speak(u);showToast({si:'🔊 කියවනවා...',ta:'🔊 படிக்கிறது...',en:'🔊 Reading aloud...'}[S.lang]);}
function rpCopy(){navigator.clipboard.writeText(rpCurrentText).then(()=>showToast({si:'✅ Copy වුනා!',ta:'✅ நகலெடுத்தது!',en:'✅ Copied!'}[S.lang]));}
function rpScroll(){const el=document.getElementById('rpContent');const pct=el.scrollTop/(el.scrollHeight-el.clientHeight)*100;document.getElementById('rpBar').style.width=Math.min(100,isNaN(pct)?0:pct)+'%';}

// ══ PANELS ══
let openPanel=null;
function togglePanel(name){const panel=document.getElementById('panel'+name.charAt(0).toUpperCase()+name.slice(1));const overlay=document.getElementById('spOverlay');if(openPanel===name){closeAllPanels();return;}closeAllPanels();openPanel=name;panel.classList.add('open');overlay.classList.add('open');if(name==='lb')loadLeaderboard();if(name==='quiz')resetQuizPanel();}
function closeAllPanels(){document.querySelectorAll('.slide-panel').forEach(p=>p.classList.remove('open'));document.getElementById('spOverlay').classList.remove('open');openPanel=null;}

// ══ QUIZ ══
let quizSessionId=null,quizAnswered=false;
function resetQuizPanel(){document.getElementById('quizBody').innerHTML=`<div style="text-align:center;color:var(--text-m);padding:20px 20px 14px;font-size:12px">${S.subject} · Grade ${S.grade}</div><button onclick="startQuiz()" style="width:100%;padding:11px;background:linear-gradient(135deg,#F59E0B,#D97706);color:#fff;border:none;border-radius:var(--r-sm);font-family:inherit;font-size:13px;font-weight:700;cursor:pointer">🚀 Start Quiz</button>`;}
async function startQuiz(){const body=document.getElementById('quizBody');body.innerHTML='<div style="text-align:center;padding:30px;color:var(--text-m)">⏳ Generating...</div>';try{const res=await fetch(`${API}/quiz/start`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_name:S.name,grade:String(S.grade),subject:S.subject,language:S.lang})});const d=await res.json();if(d.status==='success'){quizSessionId=d.session_id;renderQuizQ(d);}else body.innerHTML=`<div style="text-align:center;padding:20px;color:var(--danger)">❌ ${d.message||'Failed'}</div>`;}catch(e){body.innerHTML='<div style="text-align:center;padding:20px;color:var(--danger)">⚠️ Error</div>';}}
function renderQuizQ(d){quizAnswered=false;document.getElementById('quizBody').innerHTML=`<div class="qz-progress"><div class="qz-progress-fill" style="width:${d.progress_pct||0}%"></div></div><div class="qz-num">Question ${d.current}/${d.total} ${d.score_so_far!=null?'· Score: '+d.score_so_far:''}</div><div class="qz-question">${esc(d.question)}</div><div class="qz-opts">${(d.options||[]).map(o=>`<button class="qz-opt" onclick="answerQuiz('${esc(o.charAt(0))}',this)">${esc(o)}</button>`).join('')}</div><div id="qzExp"></div>`;}
async function answerQuiz(ans,btn){if(quizAnswered)return;quizAnswered=true;document.querySelectorAll('.qz-opt').forEach(b=>b.classList.add('locked'));try{const res=await fetch(`${API}/quiz/answer`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:quizSessionId,user_name:S.name,user_answer:ans})});const d=await res.json();btn.classList.add(d.correct?'correct':'wrong');document.getElementById('qzExp').innerHTML=`<div class="qz-exp"><strong>${d.correct?'✅':'❌'}</strong> ${esc(d.explanation||'')}</div>`;if(d.status==='completed'){setTimeout(()=>renderQuizResult(d),1200);}else{setTimeout(()=>renderQuizQ({...d}),1400);}}catch(e){quizAnswered=false;showToast('⚠️ Error');}}
function renderQuizResult(d){const pct=d.percent||0;const cls=pct>=80?'great':pct>=50?'ok':'low';document.getElementById('quizBody').innerHTML=`<div class="qz-result"><div class="qz-circle ${cls}"><span class="qz-pct" style="font-size:24px;line-height:1">${pct}%</span><span class="qz-sc" style="font-size:10px;color:var(--text-m)">${d.score}/${d.total}</span></div><h3 style="font-size:15px;margin-bottom:4px">${pct>=80?'🏆 Excellent!':pct>=50?'😊 Good job!':'😅 Keep trying!'}</h3><div class="xp-pill">⚡ +${d.xp_earned} XP</div><div style="display:flex;gap:7px;justify-content:center;margin:12px 0"><button onclick="startQuiz()" style="padding:8px 18px;background:var(--primary);color:#fff;border:none;border-radius:8px;font-family:inherit;font-size:12px;font-weight:700;cursor:pointer">🔄 Again</button><button onclick="closeAllPanels()" style="padding:8px 18px;background:var(--bg);border:1.5px solid var(--border);border-radius:8px;font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;color:var(--text-p)">Close</button></div><div class="qz-review"><div style="font-size:10px;font-weight:700;color:var(--text-m);margin-bottom:6px">REVIEW</div>${(d.answers||[]).map(a=>`<div class="qz-rev-item ${a.correct?'right':'wrong2'}"><div style="font-weight:700;margin-bottom:2px">${esc(a.question)}</div><div style="color:var(--text-m)">You: ${esc(a.user_answer)} · Answer: ${esc(a.right_answer)}</div></div>`).join('')}</div></div>`;loadProgress();}

// ══ LEADERBOARD ══
async function loadLeaderboard(){const body=document.getElementById('lbBody');body.innerHTML='<div style="text-align:center;padding:20px;color:var(--text-m)">Loading...</div>';try{const res=await fetch(`${API}/leaderboard`);const d=await res.json();const list=d.leaderboard||[];if(!list.length){body.innerHTML='<div style="text-align:center;padding:20px;color:var(--text-m)">No entries yet!</div>';return;}body.innerHTML=list.map(e=>{const rc=e.rank===1?'g':e.rank===2?'s':e.rank===3?'b':'n';const emoji=e.rank===1?'🥇':e.rank===2?'🥈':e.rank===3?'🥉':e.rank;const isYou=e.name===S.name;return`<div class="lb-row${isYou?' lb-you':''}"><div class="lb-rank ${rc}">${emoji}</div><div class="lb-name">${esc(e.name)}${isYou?' <small style="color:var(--primary)">(you)</small>':''}</div><div style="font-size:10px;color:var(--text-m)">🔥${e.streak||0}d</div><div class="lb-xp">${e.xp||0} XP</div></div>`;}).join('');}catch(e){body.innerHTML='<div style="text-align:center;padding:20px;color:var(--danger)">⚠️ Error</div>';}}

// ══ IMAGE GEN ══
async function genImage(){const inp=document.getElementById('igInp');const prompt=inp.value.trim();if(!prompt){showToast('Describe what to draw!');return;}const result=document.getElementById('igResult');result.innerHTML='<div style="padding:20px;color:var(--text-m)">🎨 Drawing...</div>';try{const res=await fetch(`${API}/generate-image`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt,subject:S.subject,language:S.lang})});const d=await res.json();if(d.status==='success'&&d.image_url){result.innerHTML=`<p style="font-size:12px;color:var(--text-p);margin-bottom:8px">${esc(d.description||'')}</p><img src="${d.image_url}" alt="Generated">`;}else{result.innerHTML=`<div style="color:var(--text-m);padding:10px">${esc(d.description||d.message||'Could not generate')}</div>`;}}catch(e){result.innerHTML='<div style="color:var(--danger);padding:10px">⚠️ Error</div>';}}

// ══ FEEDBACK ══
let fbRating=0;
function rateStar(n){fbRating=n;document.querySelectorAll('.fb-star').forEach((s,i)=>{s.classList.toggle('lit',i<n);});}
function submitFeedback(){if(!fbRating){showToast('⭐ Please rate first!');return;}const feedback={user:S.name,rating:fbRating,message:document.getElementById('fbMsg').value.trim(),time:new Date().toISOString()};console.log('Feedback:',feedback);showToast('🙏 Thank you for your feedback!');closeAllPanels();fbRating=0;document.querySelectorAll('.fb-star').forEach(s=>s.classList.remove('lit'));document.getElementById('fbMsg').value='';}

// ══ PROGRESS ══
async function loadProgress(){try{const res=await fetch(`${API}/progress/${S.name}`);const d=await res.json();if(d.status==='success')updateXpDisplay(d.progress);}catch(e){}}

// Admin shortcut
window.addEventListener('keydown',e=>{if(e.ctrlKey&&e.shiftKey&&e.key==='F2'){e.preventDefault();window.location.href='dashboard.html';}});
