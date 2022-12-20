import Link from 'next/link';
import { useRouter } from 'next/router';
import { useContext } from 'react';
import { UserContext } from '@lib/context';
import { auth } from '@lib/firebase';

// Top navbar
export default function Navbar() {
  const { user, username } = useContext(UserContext);

  const router = useRouter();

  const signOut =  () => {
    auth.signOut();
    router.reload();
  }

  return (
    <nav className="topbar-o">
      <div className='top-tab-block'>
        <li className='top-tab-block-e'>
          <Link href="/">
            <img src='Hanasu.jpeg' style={{width: '30px'}} />
          </Link>
        </li>

        {/* user is signed-in and has username */}
        {username && (
          <>
            <div className="top-tab-block-e">
              <button onClick={signOut}>Exit</button>
            </div>
            <div className='top-tab-block-e'>
              <Link href="/admin">
                <button className="btn-blue">Write Posts</button>
              </Link>
            </div>
            <div className='top-tab-block-e'>
              <Link href={`/${username}`}>
                <img src={user?.photoURL || '/hacker.png'} />
              </Link>
            </div>
          </>
        )}

        {/* user is not signed OR has not created username */}
        {!username && (
          <div className='top-tab-block'>
            <Link href="/enter">
              <button className="btn-blue">Log in</button>
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
}
