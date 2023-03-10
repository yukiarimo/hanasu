import PostFeed from '@components/PostFeed';
import Metatags from '@components/Metatags';
import { firestore, fromMillis, postToJSON } from '@lib/firebase';
import toast from 'react-hot-toast';
import { useState } from 'react';

// Max post to query per page
const LIMIT = 2;

export async function getServerSideProps(context) {
  const postsQuery = firestore
    .collectionGroup('posts')
    .where('published', '==', true)
    .orderBy('createdAt', 'desc')
    .limit(LIMIT);

  const posts = (await postsQuery.get()).docs.map(postToJSON);

  return {
    props: { posts }, // will be passed to the page component as props
  };
}

export default function Home(props) {
  const [posts, setPosts] = useState(props.posts);
  const [loading, setLoading] = useState(false);

  const [postsEnd, setPostsEnd] = useState(false);

  // Get next page in pagination query
  const getMorePosts = async () => {
    setLoading(true);
    const last = posts[posts.length - 1];

    const cursor = typeof last.createdAt === 'number' ? fromMillis(last.createdAt) : last.createdAt;

    const query = firestore
      .collectionGroup('posts')
      .where('published', '==', true)
      .orderBy('createdAt', 'desc')
      .startAfter(cursor)
      .limit(LIMIT);

    const newPosts = (await query.get()).docs.map((doc) => doc.data());

    setPosts(posts.concat(newPosts));
    toast.success('Posts loaded!');
    setLoading(false);

    if (newPosts.length < LIMIT) {
      setPostsEnd(true);
    }
  };

  return (
    <div className='block-o'>
      <Metatags title="Home Page" description="Get the latest posts on our site" />

      <div className="block-l">
        <h2>はなす</h2>
        <p>Welcome! Hanasu - is the most popular social network is the world!</p>
        <p>Created by Yuki Arimo</p>
      </div>
     
      <PostFeed posts={posts} />

      {!loading && !postsEnd && <button onClick={getMorePosts}>Next</button>}
      {postsEnd && 'Articles finished!'}
    </div>
    
  );
}
